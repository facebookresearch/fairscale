# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test AdaScale with DDP/SDP/FSDP.

    Even though it is tested here, AdaScale does NOT work with SDP/FSDP the
    same way as DDP & gradient accumulation modes, because the full
    gradients are not sent to each worker.

    So they only have a slice of the reduced gradient in FSDP's case or
    only a subset of gradients are reduced in SDP's. OTOH, each AdaScale
    work receives full local-gradient.  So the gain value computation is
    off. If they use a slice (or subset) of their local-gradient, the gain
    values they each compute will be different, which might or might not
    be helpful for training.
"""

import tempfile

import numpy as np
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

from fairscale.fair_dev.testing.golden_testing_data import adascale_test_data
from fairscale.fair_dev.testing.testing import skip_if_single_gpu
from fairscale.internal import torch_version
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import ShardedDataParallel as SDP
from fairscale.optim import OSS, AdaScale


def _dist_init(rank, world_size, tempfile_name, backend):
    url = "file://" + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _test_basic_func(rank, ddp_cls, world_size, tempfile_name, test_case):
    _dist_init(rank, world_size, tempfile_name, backend="nccl")  # Covers nccl

    model = Linear(2, 2)
    model.to("cuda")
    if ddp_cls is DDP:
        model = ddp_cls(model, device_ids=[rank])
        optim = AdaScale(SGD(model.parameters(), lr=0.1))
    elif ddp_cls is SDP:
        optim = AdaScale(OSS(model.parameters(), SGD, lr=0.1))
        model = ddp_cls(model, sharded_optimizer=optim)
    else:
        assert ddp_cls is FSDP, ddp_cls
        # Two cases:
        #    flatten=True : AdaScale wrapper must be after FSDP and it receives
        #                   a single grad tensor. It won't receive grad if
        #                   wrapped before.
        #    flatten=False: AdaScale can be both before or after FSDP.
        # So, it is better to do AdaScale after FSDP.
        model = ddp_cls(model, flatten_parameters=False)
        optim = AdaScale(SGD(model.parameters(), lr=0.1))
    if "input" in test_case:
        # single iter
        in_data = Tensor(test_case["input"][rank])
        in_data = in_data.cuda()
        out = model(in_data)
        out.sum().backward()
        if ddp_cls is DDP:
            assert np.allclose(optim.gain(), test_case["expected_gain"]), optim.gain()
            w, b = model.parameters()
            assert np.allclose(w.grad.cpu(), test_case["expected_grad"]), w.grad
            assert np.allclose(b.grad.cpu(), test_case["expected_bias_grad"]), b.grad
        optim.step()
        optim.zero_grad()
    else:
        # multiple iters
        n = len(test_case["inputs"])
        for i, in_data in enumerate(test_case["inputs"]):
            in_data = Tensor(in_data[rank]).cuda()
            out = model(in_data)
            out.sum().backward()
            if i == n - 1 and ddp_cls is DDP:
                assert np.allclose(optim.gain(), test_case["expected_gain"]), optim.gain()
                w, b = model.parameters()
                assert np.allclose(w.grad.cpu(), test_case["expected_grad"]), w.grad
                assert np.allclose(b.grad.cpu(), test_case["expected_bias_grad"]), b.grad
            optim.step()
            optim.zero_grad()

    dist.destroy_process_group()


@skip_if_single_gpu
@pytest.mark.parametrize("ddp_cls", [DDP])
@pytest.mark.parametrize("test_case", adascale_test_data)
def test_basic(ddp_cls, test_case):
    """Test adascale with DDP without gradient accumulation"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(_test_basic_func, args=(ddp_cls, world_size, temp_file_name, test_case), nprocs=world_size, join=True)


@skip_if_single_gpu
@pytest.mark.parametrize("ddp_cls", [DDP, SDP, FSDP])
@pytest.mark.parametrize("test_case", adascale_test_data[:1])
def test_basic_all_dp(ddp_cls, test_case):
    """Test adascale with DDP/SDP/FSDP with just one test case."""
    test_basic(ddp_cls, test_case)


def _test_grad_accum_func(rank, world_size, tempfile_name):
    _dist_init(rank, world_size, tempfile_name, backend="gloo")  # Covers gloo

    model = Linear(4, 2, bias=False)
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)
    with model.no_sync():
        # iter 1, input vectors are pointing dim0 and dim1
        in_data = Tensor([0.0] * 4)
        in_data[rank] = 1.0
        in_data = in_data.cuda()
        out = model(in_data)
        out.sum().backward()
    # iter 2, input vectors are pointing dim2 and dim3
    in_data = Tensor([0.0] * 4)
    in_data[rank + 2] = 1.0
    in_data = in_data.cuda()
    out = model(in_data)
    out.sum().backward()
    # since all inputs are orthogonal, the gain should be exactly 4.0.
    assert np.allclose(optim.gain(), 4.0), optim.gain()
    optim.step()
    optim.zero_grad()

    dist.destroy_process_group()


@skip_if_single_gpu
def test_grad_accum():
    """Test adascale with DDP + gradient accumulation using ddp.no_sync()"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(_test_grad_accum_func, args=(world_size, temp_file_name), nprocs=world_size, join=True)


def _test_corr_mean_func(rank, world_size, tempfile_name, test_case):
    _dist_init(rank, world_size, tempfile_name, backend="gloo")  # Covers gloo

    model = Linear(3, 1, bias=False)
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    optim = AdaScale(SGD(model.parameters(), lr=0.1))
    results = []
    last_grad = None
    for i, in_data in enumerate(test_case["inputs"]):
        # use no_sync so we can access nonreduced gradients
        with model.no_sync():
            in_data = Tensor(in_data[rank]).cuda()
            out = model(in_data)
            out.sum().backward()
            results.append(optim._compute_intra_grad_corr_mean().item())
        # sync gradients manually
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                # divide by world size
                p.grad.data.div_(world_size)
        grad = optim._gather_flat_grad()
        assert np.allclose(grad.cpu(), test_case["expected_grad"][i])
        optim.step()
        if last_grad is not None:
            # compute cosine similarity
            cos_similarity = torch.dot(grad, last_grad) / (grad.norm() * last_grad.norm())
            np.allclose(cos_similarity.cpu(), test_case["expected_cos_similarity"][i])
        last_grad = grad
        optim.zero_grad()
    assert np.allclose(results, test_case["expected_corr"]), results

    dist.destroy_process_group()


@skip_if_single_gpu
@pytest.mark.skipif(
    torch_version() < (1, 10, 0),
    reason="torch.corrcoef available only for torch 1.10 or higher",
)
def test_corr_mean():
    """
    Test _compute_intra_grad_corr_mean and _gather_flat_grad using ddp.no_sync()
    We also demonstrate how cosine similarity between consecutive gradients can be computed using _gather_flat_grad
    """
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    from fairscale.fair_dev.testing.golden_testing_data import corr_mean_test_data

    test_case = corr_mean_test_data[0]

    mp.spawn(_test_corr_mean_func, args=(world_size, temp_file_name, test_case), nprocs=world_size, join=True)
