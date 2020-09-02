# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import fairscale.optim as optim

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
DEVICE = "cuda" if torch.cuda.is_available() else torch.device("cpu")


def setup_module(module):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=BACKEND, rank=0, world_size=1)


def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)


def test_create():
    params = [torch.rand(1)]
    o = optim.OSS(params, lr=0.01)


def test_state_dict():
    x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
    o = optim.OSS([x], lr=0.1, momentum=0.9)
    x.backward()
    o.step()
    assert x == torch.tensor([0.9], device=DEVICE)
    assert o.optim.state[x]["momentum_buffer"] == torch.tensor([1.0], device=DEVICE)
    o.zero_grad()
    o.consolidate_state_dict()  # Sync state dict in between replicas - even if there are none
    state_dict = o.state_dict()

    # Check that the state dict is pytorch-compliant key wise
    assert "param_groups" in state_dict.keys()
    assert "state" in state_dict.keys()

    # Check that the pulled state is what we expect
    assert state_dict["param_groups"][0][0]["lr"] == 0.1

    # Check that the pulled state and the .param_groups attribute are in sync
    assert state_dict["param_groups"][0][0]["lr"] == o.param_groups[0]["lr"]

    # Check that it's correctly loaded
    o = optim.OSS([x], lr=0.01)
    o.load_state_dict(state_dict)
    # Check that state is correct and on proper device
    assert o.optim.state[x]["momentum_buffer"] == torch.tensor([1.0], device=DEVICE)

    # We should now be using a lr of 0.1, both within the optimizer
    # and as exposed by the .param_groups attribute
    assert o.param_groups[0]["lr"] == 0.1
    x.backward()
    o.step()
    assert x == torch.tensor([0.71], device=DEVICE)
    assert o.optim.state[x]["momentum_buffer"] == torch.tensor([1.9], device=DEVICE)

    # Check that the exposed param_groups are on the proper device
    assert o.param_groups[0]["params"][0].device == x.device


def test_lr_scheduler():
    x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
    x2 = torch.tensor([1.0], device=DEVICE, requires_grad=True)
    o = optim.OSS([x], lr=0.01)
    o2 = torch.optim.SGD([x2], lr=0.01)
    s = torch.optim.lr_scheduler.StepLR(o, 1)
    s2 = torch.optim.lr_scheduler.StepLR(o2, 1)
    for _ in range(5):
        x.backward()
        o.zero_grad()
        o.step()
        s.step()
        x2.backward()
        o2.zero_grad()
        o2.step()
        s2.step()
        assert x == x2


class SGDWithStepKWArg(torch.optim.SGD):
    def step(self, closure=None, kwarg=[]):
        super().step()
        kwarg.append(5)


def test_step_with_kwargs():
    kwarg = []
    x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
    o = optim.OSS([x], SGDWithStepKWArg, lr=0.1)
    x.backward()
    o.step(0, kwarg=kwarg)
    assert kwarg == [5]
    assert x == torch.tensor([0.9], device=DEVICE)


def test_local_state_dict():
    x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
    o = optim.OSS([x], lr=0.1)
    local_state_dict = o.local_state_dict()
    o = optim.OSS([x], lr=0.01)
    o.load_local_state_dict(local_state_dict)
    # We should now be using a lr of 0.1.
    assert o.optim.param_groups[0]["lr"] == 0.1
    assert o.param_groups[0]["lr"] == 0.1
    x.backward()
    o.step()
    assert x == torch.tensor([0.9], device=DEVICE)


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
    assert len(o.optim.param_groups) == 2


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
    world_size = min(2, torch.cuda.device_count())
    mp.spawn(run_test_step, args=(world_size,), nprocs=world_size, join=True)


def run_test_step_with_closure(rank, world_size, optimizer=None):
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
    world_size = min(2, torch.cuda.device_count())
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


def run_test_collect_shards(rank, world_size, reference_rank):
    dist_init(rank, world_size)
    device = torch.device(rank) if torch.cuda.device_count() > 1 else DEVICE

    # Run a dummy step so that the optimizer state dict exists
    batch, input_width, hidden, target_width = 3, 20, 10, 5
    target = torch.rand((batch, target_width), device=device)
    inputs = torch.rand((batch, input_width), device=device)

    model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width))
    model.to(device)

    loss_fn = torch.nn.L1Loss()
    loss_fn.to(device)

    # With SGD, Momentum is required to get a state to shard
    optimizer = optim.OSS(model.parameters(), lr=0.1, momentum=0.99)

    def closure():
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, target)
        loss.backward()
        return loss

    _ = optimizer.step(closure=closure)

    # Update the optimizer state on the reference rank
    optimizer.consolidate_state_dict(recipient_rank=reference_rank)

    # Fetch the state on the reference rank
    # - check that it has the correct size
    # - load it again
    if rank == reference_rank:
        optimizer_state_dict = optimizer.state_dict()
        assert len(optimizer_state_dict["state"]) == world_size
    else:
        optimizer_state_dict = {}

    optimizer_state_dict = optim.utils.broadcast_object(
        optimizer_state_dict, src_rank=reference_rank, group=dist.group.WORLD, dist_device=device
    )

    # Load the optimizer state dict
    optimizer.load_state_dict(optimizer_state_dict)


def test_collect_shards():
    world_size = 3
    if torch.cuda.is_available():
        world_size = min(world_size, torch.cuda.device_count())
    reference_rank = 0

    mp.spawn(
        run_test_collect_shards, args=(world_size, reference_rank), nprocs=world_size, join=True,
    )
