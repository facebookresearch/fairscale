# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test AdaScale with OSS. """

from statistics import mean
import tempfile

import numpy as np
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

from fairscale.optim import OSS, AdaScale
from fairscale.utils.testing import skip_if_single_gpu


def _dist_init(rank, world_size, tempfile_name, backend):
    url = "file://" + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _test_basic_func(rank, world_size, tempfile_name, test_case, oss, model=None):
    _dist_init(rank, world_size, tempfile_name, backend="nccl")

    if model is None:
        model = Linear(2, 2, bias=False)
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    if oss:
        optim = AdaScale(OSS(model.parameters(), SGD, lr=0.1))
    else:
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
    if "expected_mean_weight" in test_case:
        mean_weight = mean([model.module[i].weight.data.mean().item() for i in range(4)])
        assert np.allclose(mean_weight, test_case["expected_mean_weight"]), mean_weight

    dist.destroy_process_group()


# IMPORTANT: make sure these test_cases values are sync'ed with the non-DDP
# test in test_single_node_adascale.py. This way, we make sure gradient accumulation
# works exactly like that in DDP & DDP+OSS.
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
    """Test adascale with DDP + OSS with trivial model"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(_test_basic_func, args=(world_size, temp_file_name, test_case), nprocs=world_size, join=True)


@skip_if_single_gpu
@pytest.mark.parametrize("oss", [True, False])
def test_sequential(oss):
    """Test adascale with DDP + OSS with a sequential model"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    # Run multiple iterations, check the gain for both oss and non-oss cases.
    test_case = {
        "inputs": [[[1.0, 0], [0, 1.0]], [[0, 1.0], [1.0, 0]]],
        "expected_gain": 1.0335265132125744,
        "expected_mean_weight": 52.92657661437988,
    }

    # The model.
    model = Sequential(
        Linear(2, 3, bias=False), Linear(3, 4, bias=False), Linear(4, 5, bias=False), Linear(5, 6, bias=False)
    )

    # Weights need to be fixed for deterministic gain.
    model[0].weight.data.copy_(Tensor(range(6)).reshape(3, 2) / mean(range(6)))
    model[1].weight.data.copy_(Tensor(range(12)).reshape(4, 3) / mean(range(12)))
    model[2].weight.data.copy_(Tensor(range(20)).reshape(5, 4) / mean(range(20)))
    model[3].weight.data.copy_(Tensor(range(30)).reshape(6, 5) / mean(range(30)))

    mp.spawn(_test_basic_func, args=(world_size, temp_file_name, test_case, oss, model), nprocs=world_size, join=True)
