# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test AdaScale with DDP. """

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

from fairscale.optim import AdaScale

skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs are required")


def _dist_init(rank, world_size, tempfile_name, backend):
    url = "file://" + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _test_basic_func(rank, world_size, tempfile_name, test_case):
    _dist_init(rank, world_size, tempfile_name, backend="nccl")  # Covers nccl

    model = Linear(2, 2, bias=False)
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    optim = AdaScale(SGD(model.parameters(), lr=0.1))
    if "input" in test_case:
        # single iter
        in_data = Tensor(test_case["input"][rank])
        in_data = in_data.cuda()
        out = model(in_data)
        out.sum().backward()
        assert np.allclose(optim.gain(), test_case["expected_gain"]), optim.gain()
        optim.step()
        optim.zero_grad()
    else:
        # multiple iters
        for in_data in test_case["inputs"]:
            in_data = Tensor(in_data[rank]).cuda()
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
        assert np.allclose(optim.gain(), test_case["expected_gain"]), optim.gain()

    dist.destroy_process_group()


# IMPORTANT: make sure these test_cases values are sync'ed with the non-DDP
# test in test_single_node_adascale.py. This way, we make sure gradient accumulation
# works exactly like that in DDP.
@skip_if_single_gpu
@pytest.mark.parametrize(
    "test_case",
    [
        # "input" value is a list of input tensors for rank 0 and rank 1.
        {"input": [[1.0, 0], [0, 1.0]], "expected_gain": 2.0},
        {"input": [[1.0, 1.0], [1.0, 1.0]], "expected_gain": 1.0000001249999846},
        {"input": [[-1.0, 1.0], [1.0, -1.0]], "expected_gain": 2.0},
        {"input": [[1.0, 4.0], [5.0, 0.5]], "expected_gain": 1.5022222222222221},
        {"input": [[-0.2, 3.0], [5.0, 0.5]], "expected_gain": 1.9433267229211089},
        # "inputs" to trigger multiple iteration tests, which make sure the
        # smoothing factor calculation is also covered.
        {"inputs": [[[-0.2, 3.3], [5.2, 0.7]], [[1.0, 4.0], [3.1, 0.1]]], "expected_gain": 1.744159431359284},
    ],
)
def test_basic(test_case):
    """Test adascale with DDP without gradient accumulation"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(_test_basic_func, args=(world_size, temp_file_name, test_case), nprocs=world_size, join=True)


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
