# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch

from fairscale.experimental.wgit.signal_sparsity import SignalSparsity

TRIALS = 5
TENSOR_DIM_LOWER = 40
TENSOR_DIM_UPPER = 100


def test_dense_to_sst_percent():

    for _ in range(TRIALS):
        # random tensor creation
        size1 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        size2 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        tensor = torch.randn(size1, size2)

        # at no sparsity: we expect almost total reconstruction
        sparser_2d = SignalSparsity(sst_top_k_percent=1.0)
        sst = sparser_2d.dense_to_sst(tensor)
        w_prime = torch.fft.ifft(sst)
        assert torch.isclose(w_prime.real, tensor, atol=1e-3).sum() == tensor.numel()

        # tests at random sparsity level
        topk_percent = random.uniform(0, 1)  # random topk_percentage
        sparser_2d = SignalSparsity(sst_top_k_percent=topk_percent)
        abs_sst = torch.abs(sparser_2d.dense_to_sst(tensor))
        # calculate the sparsity of the returned tensor
        sps = 1 - abs_sst.count_nonzero() / abs_sst.numel()
        # assert that the sparsity of the returned sst is close to the target sparsity
        assert torch.isclose(sps, torch.tensor(1 - topk_percent, device=sps.device), atol=8e-2)


def test_dense_to_sst_elements():
    def test_dense_to_sst_topk_elem(tensor, topk_elem, dim):
        sparser_2d = SignalSparsity(sst_top_k_element=topk_elem, sst_top_k_dim=dim)
        sst = sparser_2d.dense_to_sst(tensor)

        # Check to verify if the topk is returning k elements along the selected dim.
        if dim is not None:
            assert all((torch.abs(sst) > 0.0).sum(dim) == topk_elem)
            # verify if the topk values of the returned sst are the correct topk values
            def_v, _ = torch.abs(torch.fft.fft(tensor)).topk(k=topk_elem, dim=dim)
            # get the topk values from default operations
            sst_v, _ = torch.abs(sst).topk(k=topk_elem, dim=dim)
            assert all((def_v == sst_v).flatten())
        else:
            assert (torch.abs(sst) > 0.0).sum() == topk_elem

        # NOTE: In lots of cases, the topk indices do not line up due to the
        # presence of duplicate values in the tensors.

    for _ in range(TRIALS):
        # random tensor creation
        size1 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        size2 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        tensor = torch.randn(size1, size2)

        # Test topk_element along possible dims for 2D tensors
        for dim in range(2):
            topk_elem = random.randrange(0, tensor.shape[dim])
            test_dense_to_sst_topk_elem(tensor, topk_elem, dim=dim)

        # # Test topk_element when dim = None
        topk_elem = random.randrange(0, tensor.numel())
        test_dense_to_sst_topk_elem(tensor, topk_elem, dim=None)
