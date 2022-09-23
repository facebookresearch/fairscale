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

from fairscale.fair_dev.testing.golden_testing_data import adascale_test_data
from fairscale.fair_dev.testing.testing import skip_if_single_gpu
from fairscale.optim import OSS, AdaScale, AdaScaleWrapper


def _dist_init(rank, world_size, tempfile_name, backend):
    url = "file://" + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _test_basic_func(rank, world_size, tempfile_name, test_case, oss, model=None):
    _dist_init(rank, world_size, tempfile_name, backend="nccl")

    if model is None:
        model = Linear(2, 2)
        model.bias.data.fill_(0.0)

    model.to("cuda")
    model = DDP(model, device_ids=[rank])

    assert oss in ["none", "ada-oss", "wrapper-oss", "oss-wrapper"]
    if oss == "ada-oss":
        optim = AdaScale(OSS(model.parameters(), SGD, lr=0.1))
    elif oss == "wrapper-oss":
        optim = AdaScaleWrapper(model.parameters(), optim_cls=OSS, optim=SGD, lr=0.1)
    elif oss == "oss-wrapper":
        optim = OSS(model.parameters(), AdaScaleWrapper, optim_cls=SGD, lr=0.1)
    else:
        assert oss == "none"
        optim = AdaScale(SGD(model.parameters(), lr=0.1))

    if "input" in test_case:
        inputs = [test_case["input"]]
    else:
        inputs = test_case["inputs"]

    for in_data in inputs:
        in_data = Tensor(in_data[rank]).cuda()
        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    if "expected_gain" in test_case:
        assert np.allclose(optim.gain(), test_case["expected_gain"]), "{} vs {}".format(
            optim.gain(), test_case["expected_gain"]
        )

    if "expected_mean_weight" in test_case:
        mean_weight = mean([model.module[i].weight.data.mean().item() for i in range(4)])
        assert np.allclose(mean_weight, test_case["expected_mean_weight"]), mean_weight

    dist.destroy_process_group()


@skip_if_single_gpu
@pytest.mark.parametrize("test_case", adascale_test_data)
def test_basic(test_case):
    """Test adascale with DDP + OSS with trivial model"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(_test_basic_func, args=(world_size, temp_file_name, test_case, "ada-oss"), nprocs=world_size, join=True)


@skip_if_single_gpu
@pytest.mark.parametrize("oss", ["none", "ada-oss", "wrapper-oss", "oss-wrapper"])
def test_sequential(oss):
    """Test adascale with DDP + OSS with a sequential model"""
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    # Run multiple iterations, check the gain for both oss and non-oss cases.
    #
    # The inputs are picked arbitrarily. I used vectors that are orthogonal.
    #
    # The gain and mean_weight values are recorded from my testing and used here
    # to ensure their value is unchanged from commit to commit unless we can
    # explain why.
    test_case = {
        "inputs": [[[1.0, 0], [0, 1.0]], [[0, 1.0], [1.0, 0]]],
        "expected_gain": 1.0335265132125744,
        "expected_mean_weight": 52.92657661437988,
    }

    if oss == "oss-wrapper":
        # When OSS wraps AdaScale, the training is numerically different
        # and it exists only to enable future research. So we don't check
        # the gain (OSS doesn't have a gain() function, different rank's
        # gains are different). We just ensure the mean_weight is expected.
        del test_case["expected_gain"]
        test_case["expected_mean_weight"] = 94.93386840820312

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
