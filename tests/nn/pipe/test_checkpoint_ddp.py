# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Test checkpoint and PyTorch DDP interactions.


import os
import random
import tempfile

import numpy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import Linear, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from fairscale.fair_dev.testing.testing import skip_if_no_cuda, skip_if_single_gpu
from fairscale.nn.pipe.checkpoint import Checkpointing, Function, TensorOrTensors
from fairscale.nn.pipe.microbatch import Batch

# This test is mainly for checking pytorch & checkpointing behavior. pipe's checkpointing
# code is tested already in another file. Therefore, we can run this test less frequently.
# We use getpid() in case random is seeded to be deterministic.
run_test = False
if os.getpid() % 100 == 42:
    run_test = True

skip_if_not_needed = pytest.mark.skipif(not run_test, reason="Skipping due to test frequency")


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def pipe_checkpoint(function: Function, input: TensorOrTensors) -> TensorOrTensors:
    """Makes a checkpoint with a simple interface like
    :func:`torch.utils.checkpoint.checkpoint`. It's only used to test or debug
    :class:`Checkpoint` and :class:`Recompute` without boilerplate.
    """
    batch = Batch(input, 0)

    chk = Checkpointing(function, batch)
    batch = chk.checkpoint()
    chk.recompute(batch)

    return batch.tensor_or_tensors


def basic(rank, checkpoint):
    # get the model, wrap with DDP and fwd, bwd.
    set_random_seed(31415)
    model = Sequential(Linear(2000, 2000), Linear(2000, 2000))
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    input_tensor = torch.rand((64, 2000)).cuda()
    input_tensor.requires_grad = True
    output_tensor = checkpoint(model, input_tensor)
    for p in model.parameters():
        assert p.grad is None
    output_tensor.sum().backward()
    norm = 0.0
    for p in model.parameters():
        assert p.grad is not None
        norm += p.grad.norm().item()
    assert numpy.allclose(norm, 78053.52978515625), norm


def weight_sharing(rank, checkpoint):
    # get the model, wrap with DDP and fwd, bwd.
    set_random_seed(31415)
    l1 = Linear(2000, 2000)
    l2 = Linear(2000, 2000)
    l1.weight = l2.weight
    model = Sequential(l1, l2)
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    input_tensor = torch.rand((64, 2000)).cuda()
    input_tensor.requires_grad = True
    output_tensor = checkpoint(model, input_tensor)
    output_tensor.sum().backward()
    norm = 0.0
    for p in model.parameters():
        assert p.grad is not None
        norm += p.grad.norm().item()
    assert numpy.allclose(norm, 57004.34228515625), norm


def checkpoint_half(rank, checkpoint):
    # get the model, wrap with DDP and fwd, bwd.
    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.l1 = Linear(2000, 2000)
            self.l2 = Linear(2000, 2000)

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x)
            return x

    set_random_seed(31415)
    model = M()
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    input_tensor = torch.rand((64, 2000)).cuda()
    output_tensor = model(input_tensor)
    output_tensor.sum().backward()
    norm = 0.0
    for p in model.parameters():
        assert p.grad is not None
        norm += p.grad.norm().item()
    assert numpy.allclose(norm, 78053.52978515625), norm


def unused_param(rank, checkpoint):
    # get the model, wrap with DDP and fwd, bwd.
    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            # The size 2000 is important. Without bigger size, it doesn't trigger the RuntimeError!
            self.l1 = Linear(2000, 2000)
            self.l2 = Linear(2000, 2000)

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x)
            return x

    model = M()
    model.to("cuda")
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    input_tensor = torch.rand((64, 2000)).cuda()
    output_tensor = model(input_tensor)
    try:
        output_tensor.sum().backward()
    except RuntimeError:
        return
    assert 0


def checkpoint_twice(rank, checkpoint):
    # get the model, wrap with DDP and fwd, bwd.
    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            # The size 2000 is important. Without bigger size, it doesn't trigger the RuntimeError!
            self.l1 = Linear(2000, 2000)
            self.l2 = Linear(2000, 2000)

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x)
            x = checkpoint(self.l2, x)
            return x

    model = M()
    model.to("cuda")
    model = DDP(model, device_ids=[rank])
    input_tensor = torch.rand((64, 2000)).cuda()
    output_tensor = model(input_tensor)
    try:
        output_tensor.sum().backward()
    except RuntimeError:
        return
    assert 0


def run(rank, world_size, temp_file_name, checkpoint, test_func):
    # setup
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=dist.Backend.NCCL, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # actual test
    test_func(rank, checkpoint)

    # cleanup
    dist.destroy_process_group()


@skip_if_not_needed
@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("checkpoint", [pipe_checkpoint, torch_checkpoint])
@pytest.mark.parametrize("test_func", [basic, weight_sharing, checkpoint_half, unused_param, checkpoint_twice])
def test_basic_ddp(checkpoint, test_func):
    temp_file_name = tempfile.mkstemp()[1]
    world_size = 2
    mp.spawn(run, args=(world_size, temp_file_name, checkpoint, test_func), nprocs=world_size, join=True)
