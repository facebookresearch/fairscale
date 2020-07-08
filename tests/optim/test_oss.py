# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import fairscale.optim as optim

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def setup_module(module):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=0, world_size=1)


def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def test_create():
    params = [torch.rand(1)]
    o = optim.OSS(params, lr=0.01)


@skip_if_no_cuda
def test_state_dict():
    x = torch.tensor([1.0], device="cuda", requires_grad=True)
    o = optim.OSS([x], lr=0.1)
    state_dict = o.state_dict()
    o = optim.OSS([x], lr=0.01)
    o.load_state_dict(state_dict)
    # We should now be using a lr of 0.1.
    x.backward()
    o.step()
    assert x == torch.tensor([0.9], device="cuda")


def run_test_add_param_group(rank, world_size):
    dist_init(rank, world_size)
    params = []
    for size in [4, 5, 2, 6, 4]:
        params.append(torch.rand(size, 1))
    o = optim.OSS(params, lr=0.1)
    assert len(o.param_groups) == 1
    o.add_param_group({"params": [torch.rand(3, 1)]})
    assert len(o.param_groups) == 2
    # Verify that added group is added to the correct partition making all have 8 elements.
    assert sum([x.numel() for g in o.optim.param_groups for x in g["params"]]) == 8
    if rank == 1:
        len(o.optim.param_groups) == 2
    else:
        len(o.optim.param_groups) == 1


def test_add_param_group():
    world_size = 3
    mp.spawn(run_test_add_param_group, args=(world_size,), nprocs=world_size, join=True)


def run_test_zero_grad(rank, world_size):
    dist_init(rank, world_size)
    x = torch.rand(1)
    m = torch.nn.Linear(1, 1)
    o = optim.OSS(m.parameters(), lr=0.1)
    y = m(x)
    y.backward(x)
    assert m.weight.grad
    assert m.bias.grad
    o.zero_grad()
    assert not m.weight.grad
    assert not m.bias.grad


@skip_if_no_cuda
def test_zero_grad():
    world_size = 2
    mp.spawn(run_test_zero_grad, args=(world_size,), nprocs=world_size, join=True)


def run_test_step(rank, world_size):
    dist_init(rank, world_size)
    x = torch.tensor([float(rank + 1)], device=rank)
    m = torch.nn.Linear(1, 1)
    m.weight.data = torch.tensor([[1.0]])
    m.bias.data = torch.tensor([2.0])
    m.to(rank)
    o = optim.OSS(m.parameters(), lr=0.1)
    y = m(x)
    y.backward(x)
    for p in m.parameters():
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
        p.grad.data /= world_size
    o.step()
    assert m.weight == torch.tensor([[0.75]], device=rank)
    assert m.bias == torch.tensor([1.85], device=rank)


@skip_if_no_cuda
def test_step():
    world_size = 2
    mp.spawn(run_test_step, args=(world_size,), nprocs=world_size, join=True)


def run_test_step_with_closure(rank, world_size):
    dist_init(rank, world_size)
    x_val = rank + 1
    weight = 1.0
    bias = 2.0
    error = 1.0
    target = torch.tensor([x_val * weight + bias + error], device=rank)
    loss_fn = torch.nn.L1Loss()

    x = torch.tensor([float(x_val)], device=rank)
    m = torch.nn.Linear(1, 1)
    m.weight.data = torch.tensor([[weight]])
    m.bias.data = torch.tensor([bias])
    m.to(rank)
    o = optim.OSS(m.parameters(), lr=0.1)
    y = m(x)
    y.backward(x)
    for p in m.parameters():
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
        p.grad.data /= world_size

    def closure():
        o.zero_grad()
        output = m(x)
        loss = loss_fn(output, target)
        loss.backward()
        return loss

    loss = o.step(closure=closure)

    assert loss == torch.tensor(error, device=rank)
    assert m.weight == torch.tensor([[1.1]], device=rank)
    assert m.bias == torch.tensor([2.1], device=rank)


@skip_if_no_cuda
def test_step_with_closure():
    world_size = 2
    mp.spawn(run_test_step_with_closure, args=(world_size,), nprocs=world_size, join=True)


def run_test_sharding(rank, world_size):
    dist_init(rank, world_size)
    params = []
    for size in [5, 4, 2, 6, 4, 3]:
        params.append(torch.rand(size, 1))
    o = optim.OSS(params, lr=0.1)
    assert sum([x.numel() for x in o.optim.param_groups[0]["params"]]) == 8


def test_sharding():
    world_size = 3
    mp.spawn(run_test_sharding, args=(world_size,), nprocs=world_size, join=True)
