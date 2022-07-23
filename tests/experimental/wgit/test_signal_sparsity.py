# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

from fairscale.experimental.wgit.signal_sparsity import SignalSparsity, _get_k_for_topk, _top_k_total_size

TRIALS = 5
TENSOR_DIM_LOWER = 40
TENSOR_DIM_UPPER = 100
DIM_LIST = [None, 0, 1]
SPARSITY_ATOL = 8e-2
FFT_RECONS_ATOL = 2e-3


def test_validate_conf():
    """Tests the config validation for the Signal Sparsity class."""

    def kwargs(vals_list):
        """Maps the values in input vals_list to the keys in arg_key_list"""

        arg_key_list = [
            "sst_top_k_element",
            "sst_top_k_percent",
            "sst_top_k_dim",
            "dst_top_k_element",
            "dst_top_k_percent",
            "dst_top_k_dim",
        ]
        return dict(zip(arg_key_list, vals_list))

    # Validate assertion error is raised when, either
    # 1. both of sst (or dst) percent and element is set to values (not None).
    # 2. both of sst (or dst) percent and element is set to None.
    element = random.randint(0, TENSOR_DIM_UPPER)
    percent = random.uniform(0, 100)
    dim = random.randint(-1, 3)

    pytest.raises(AssertionError, SignalSparsity, **kwargs([element, percent, dim, element, None, dim]))
    pytest.raises(AssertionError, SignalSparsity, **kwargs([element, None, dim, element, percent, dim]))
    pytest.raises(AssertionError, SignalSparsity, **kwargs([None, None, dim, element, None, dim]))
    pytest.raises(AssertionError, SignalSparsity, **kwargs([element, None, dim, None, None, dim]))


def test_sst_sparsity_level():
    """Tests the dense_to_sst method of SignalSparsity in terms of generating SST at the correct
    sparsity level.
    """

    def get_sst(sst_top_k_percent, dst_top_k_percent=100):
        sparser_2d = SignalSparsity(sst_top_k_percent=sst_top_k_percent, dst_top_k_percent=100)
        sst = sparser_2d.dense_to_sst(tensor)
        return sst

    for _ in range(TRIALS):
        # random tensor creation
        size1 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        size2 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        tensor = torch.randn(size1, size2)

        # at no sparsity: we expect almost total reconstruction
        sst = get_sst(100)
        w_prime = torch.fft.ifft(sst)
        assert torch.isclose(w_prime.real, tensor, atol=FFT_RECONS_ATOL).sum() == tensor.numel()

        # tests at random sparsity level
        topk_percent = random.uniform(0, 100)
        sst = get_sst(topk_percent)
        abs_sst = torch.abs(sst)

        # sparsity of the returned tensor
        sps = 1 - abs_sst.count_nonzero() / abs_sst.numel()

        # assert that the sparsity of the returned sst is close to the target sparsity
        target_sps = torch.tensor(100 - topk_percent, device=sps.device) / 100
        assert torch.isclose(sps, target_sps, atol=SPARSITY_ATOL)  # sparsity values can be coarsely close.


def test_dense_to_sst_topk_values():
    """Tests the dense_to_sst method of SignalSparsity in terms of correct top-k value extraction in the SST."""

    def test_dense_to_sst_topk_elem(tensor, topk_elem, topk_percent, dim):
        sparser_2d = SignalSparsity(
            sst_top_k_element=topk_elem, sst_top_k_percent=topk_percent, sst_top_k_dim=dim, dst_top_k_percent=100
        )
        sst = sparser_2d.dense_to_sst(tensor)

        # when testing with percentage, get the corresponding top-k number.
        if topk_percent is not None:
            topk_elem = _get_k_for_topk(topk_percent, topk_elem, _top_k_total_size(tensor, dim))

        # the output of the dense_to_sst is compared with default transfrom from pytorch
        # NOTE: the following block of code depends on whether we are using the absolute value
        # (complex magnitudes) or only the real part for selecting the topk in
        # dense_to_sst method. Change in this criterion should result in a change below.

        # get the topk value tensor from default transform and topk operations
        default_freq = torch.fft.fft(tensor)
        default_real_freq = torch.abs(default_freq.real)  # modify default_freq.real -> torch.abs(default_freq)

        if dim is None:
            assert (torch.abs(sst) > 0.0).sum() == topk_elem
            # compare sst topk with topk gathered from default computation
            _, i = default_real_freq.flatten().topk(topk_elem)
            assert all(sst.flatten()[i] == default_freq.flatten()[i])
        else:
            # verify if sst only has top-k values in the right dim
            sst_nonzero_along_dim = (torch.abs(sst) > 0.0).sum(dim)

            # when topk_elem is 0, we explicitly ensure that SignalSparsity class ensures we get top-1 element.
            assert all(sst_nonzero_along_dim == topk_elem) if (topk_elem != 0) else all(sst_nonzero_along_dim == 1)

            # use the topk values' indices from the default_real_freq and gather default_freq vals
            # The  values from the freq tensor as per this index should match the same index values
            # from the returned SST tensor.
            _, idx = default_real_freq.topk(k=topk_elem, dim=dim)
            def_vals = default_freq.gather(dim, idx)

            sst_vals = sst.gather(dim, idx)
            assert all((def_vals == sst_vals).flatten())

    for _ in range(TRIALS):
        # random tensor creation
        size1 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        size2 = random.randint(TENSOR_DIM_LOWER, TENSOR_DIM_UPPER)
        tensor = torch.randn(size1, size2)

        # Test topk_element along possible dims for 2D tensors
        for dim in DIM_LIST:
            # with topk element
            max_elem_in_dim = tensor.numel() if (dim is None) else tensor.shape[dim]
            topk_elem = random.randrange(0, max_elem_in_dim)  # sample a k for topk
            test_dense_to_sst_topk_elem(tensor, topk_elem, None, dim=dim)

            # with topk percent
            topk_percent = random.uniform(0, 100)
            test_dense_to_sst_topk_elem(tensor, None, topk_percent, dim=dim)
