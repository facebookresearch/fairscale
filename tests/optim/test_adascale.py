# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


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

skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")


def test_basic_cpu():
    """Test single batch behavior on CPU"""
    model = Linear(2, 2, bias=False)
    model.weight.data[0, 0] = 0.1
    model.weight.data[0, 1] = 0.2
    model.weight.data[1, 0] = 0.3
    model.weight.data[1, 1] = 0.4
    try:
        optim = AdaScale(SGD(model.parameters(), lr=0.1))
    except RuntimeError:
        return
    assert False, "Single batch AdaScale should not be suppported"


def test_loss_accum_cpu():
    """Test the loss accumulation behavior on CPU

    Loss accumulation is NOT SUPPORTED. This test shows that it does not work.
    """
    model = Linear(2, 2, bias=False)
    model.weight.data[0, 0] = 0.1
    model.weight.data[0, 1] = 0.2
    model.weight.data[1, 0] = 0.3
    model.weight.data[1, 1] = 0.4
    # num_gradients_to_accumulate value doesn't matter
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=123)
    # data 1
    in_data = Tensor([0.0, 1.0])
    loss = model(in_data).sum()
    # data 2
    in_data = Tensor([1.0, 0.0])
    loss += model(in_data).sum()
    # data 3
    in_data = Tensor([1.0, 2.0])
    loss += model(in_data).sum()
    # backward, but gradient is only produced once by the autograd engine.
    loss.backward()
    # therefore, the gain will always be 1, which render adascale as noop.
    optim.step()
    assert np.allclose(optim.gain(), 1.0), optim.gain()


def test_grad_accum_cpu():
    """Test the basic functionality on CPU"""
    model = Linear(2, 2, bias=False)
    model.weight.data[0, 0] = 0.1
    model.weight.data[0, 1] = 0.2
    model.weight.data[1, 0] = 0.3
    model.weight.data[1, 1] = 0.4
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)
    for expected_gain in [2.0, 2.0]:  # do 2 iterations
        # grad pass 1
        in_data = Tensor([0.0, 1.0])
        out = model(in_data)
        out.sum().backward()
        # grad pass 2
        in_data = Tensor([1.0, 0.0])
        out = model(in_data)
        out.sum().backward()
        # stepping it. Note that if we did more than 2 passes, AdaScale won't be able to
        # check it for now. The result will just be wrong if that's the case.
        assert np.allclose(optim.gain(), expected_gain), optim.gain()
        optim.step()
        optim.zero_grad()


def test_state_checkpointing_ddp():
    # TODO:
    # run without checkpointing
    # run with checkpointing in the middle
    # assert the results are the same
    pass


def _dist_init(rank, world_size, tempfile_name, backend):
    url = "file://" + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _test_ddp_func(rank, world_size, tempfile_name):
    _dist_init(rank, world_size, tempfile_name, backend="nccl")

    model = Linear(2, 2, bias=False)
    model.weight.data[0, 0] = 0.1
    model.weight.data[0, 1] = 0.2
    model.weight.data[1, 0] = 0.3
    model.weight.data[1, 1] = 0.4
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    optim = AdaScale(SGD(model.parameters(), lr=0.1))
    # iter 1
    in_data = Tensor([0.0, 0.0])
    in_data[rank] = 1.0
    in_data = in_data.cuda()
    out = model(in_data)
    out.sum().backward()
    assert np.allclose(optim.gain(), 2.0), optim.gain()
    optim.step()
    optim.zero_grad()

    dist.destroy_process_group()


@skip_if_single_gpu
def test_ddp():
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(_test_ddp_func, args=(world_size, temp_file_name), nprocs=world_size, join=True)


@skip_if_single_gpu
def test_grad_accum_ddp():
    # TODO: test like ddp but with model.no_sync()
    pass
