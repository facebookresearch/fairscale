# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


import copy
from math import inf
import tempfile
from typing import Any, Dict, Type, cast
import unittest

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.fair_dev.testing.testing import (
    check_same_model_params,
    check_same_models_across_ranks,
    skip_if_no_cuda,
    skip_if_py39_no_cuda,
    skip_if_single_gpu,
)
from fairscale.internal import torch_version
import fairscale.optim as optim

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
DEVICE = "cuda" if torch.cuda.is_available() else torch.device("cpu")
RECIPIENT_RANK = 1


def dist_init(rank, world_size, tempfile_name, backend=BACKEND):
    url = "file://" + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)


def sync_object_ranks(something_to_sync: Any, reference_rank: int, device: torch.device) -> Any:
    package = [something_to_sync]
    dist.broadcast_object_list(package, src=reference_rank, group=dist.group.WORLD)
    package_sync = package[0]
    return package_sync


class TestSingleRank(unittest.TestCase):
    """
    All the following tests do not check for inter-process communication
    """

    def setUp(self):
        dist_init(0, 1, tempfile.mkstemp()[1])

    def tearDown(self):
        torch.distributed.destroy_process_group()

    def test_create(self):
        params = [torch.rand(1)]
        o = optim.OSS(params, lr=0.01)

    def test_state_dict(self):
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

        # Check that the pulled state is what we expect, and that we have all the expected keys
        assert state_dict["param_groups"][0]["lr"] == 0.1
        assert state_dict["param_groups"][0]["momentum"] == 0.9
        assert not state_dict["param_groups"][0]["nesterov"]
        assert state_dict["param_groups"][0]["weight_decay"] == 0.0
        assert state_dict["param_groups"][0]["dampening"] == 0.0

        # Check that the pulled state and the .param_groups attribute are in sync
        for k in state_dict["param_groups"][0].keys():
            if k != "params":
                assert state_dict["param_groups"][0][k] == o.param_groups[0][k]

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

    def test_lr_scheduler(self):
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

    def test_step_with_kwargs(self):
        class SGDWithStepKWArg(torch.optim.SGD):
            def step(self, closure=None, kwarg=[]):
                super().step()
                kwarg.append(5)

        kwarg = []
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = optim.OSS([x], SGDWithStepKWArg, lr=0.1)
        x.backward()
        o.step(0, kwarg=kwarg)
        assert kwarg == [5]
        assert x == torch.tensor([0.9], device=DEVICE)

    @skip_if_no_cuda
    def test_device_change(self):
        x = torch.nn.Linear(1, 1).to("cpu")
        o = optim.OSS(x.parameters(), torch.optim.SGD, lr=0.1)

        # Move the model to device after OSS was constructed
        x.to(DEVICE)
        x(torch.zeros(1, device=DEVICE)).backward()

        # Check that OSS detects that the device changed
        o.step()

        # Check that the default device has been updated
        assert o._default_device.type == DEVICE

    def test_step_with_extra_inner_key(self):
        class SGDWithNewKey(torch.optim.SGD):
            # Dummy optimizer which adds a new key to the param groups
            def step(self, closure=None):
                super().step()
                self.param_groups[0]["new_key"] = 0.1

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = optim.OSS([x], SGDWithNewKey, lr=0.1)
        x.backward()
        o.step()
        assert o.param_groups[0]["new_key"] == 0.1
        assert x == torch.tensor([0.9], device=DEVICE)

    def test_step_without_closure(self):
        class SGDWithoutClosure(torch.optim.SGD):
            def step(self):
                return super().step()

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = optim.OSS([x], SGDWithoutClosure, lr=0.1)
        x.backward()
        o.step()
        assert x == torch.tensor([0.9], device=DEVICE)

    def test_implicit_local_state_dict(self):
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = optim.OSS([x], lr=0.1)
        with pytest.raises(RuntimeError):
            _ = o.state_dict()


def run_test_add_param_group(rank, world_size, tempfile_name):
    dist_init(rank, world_size, tempfile_name)

    # Test with all parameters trainable to begin with
    def all_trainable():
        params = []
        sizes = [9, 7, 5, 3]
        sizes_world = sizes * world_size
        for size in sizes_world[:-1]:
            params.append(torch.rand(size, 1))

        # Make sure that the params are trainable, enforces size-based partitioning
        for p in params:
            p.requires_grad = True

        o = optim.OSS(params, lr=0.1)

        assert len(o.param_groups) == 1
        o.add_param_group({"params": [torch.rand(3, 1)]})

        assert len(o.param_groups) == 2

        # Verify that added group is added to the correct partition making all have the same number of elements
        assert sum([x.numel() for g in o.optim.param_groups for x in g["params"]]) == sum(sizes)
        assert len(o.optim.param_groups) == 2

    # Test a pathological config with a first big non-trainable param
    def some_trainable():
        params = []
        for size in [100, 3, 5, 2, 6, 4]:
            params.append(torch.rand(size, 1))

        # Make sure that the params are trainable, enforces size-based partitioning
        for p in params[1:]:
            p.requires_grad = True

        o = optim.OSS(params, lr=0.1)

        assert len(o.param_groups) == 1
        o.add_param_group({"params": [torch.rand(3, 1)]})

        assert len(o.param_groups) == 2
        assert len(o.optim.param_groups) == 2

    all_trainable()
    some_trainable()

    dist.destroy_process_group()


def test_add_param_group():
    world_size = 4
    if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
        world_size = min(world_size, torch.cuda.device_count())

    mp.spawn(run_test_add_param_group, args=(world_size, tempfile.mkstemp()[1]), nprocs=world_size, join=True)


def run_test_zero_grad(rank, world_size, tempfile_name):
    dist_init(rank, world_size, tempfile_name)
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

    dist.destroy_process_group()


def test_zero_grad():
    world_size = 2
    if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
        world_size = min(world_size, torch.cuda.device_count())

    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_test_zero_grad, args=(world_size, temp_file_name), nprocs=world_size, join=True)


def run_test_empty_shard(rank, world_size, tempfile_name, backend):
    dist_init(rank, world_size, tempfile_name, backend=backend)
    m = torch.nn.Linear(1, 1)
    x = torch.rand(20, 1)

    if torch.cuda.is_available():
        m = m.to(rank)
        x = x.to(rank)

    o = optim.OSS(m.parameters(), lr=0.1)
    y = m(x).sum()
    y.backward()
    o.step()

    dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_empty_shard(backend):
    world_size = 4
    if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
        world_size = min(world_size, torch.cuda.device_count())
    if world_size == 1 or (backend == "nccl" and not torch.cuda.is_available()):
        pytest.skip("Not enough GPUs to test with NCCL, or CUDA not present")
    mp.spawn(run_test_empty_shard, args=(world_size, tempfile.mkstemp()[1], backend), nprocs=world_size, join=True)


def run_test_step(rank, world_size, tempfile_name):
    dist_init(rank, world_size, tempfile_name, backend="gloo")
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
    assert m.weight == torch.tensor([[0.75]], device=rank), f"{rank}: {m.weight.item()}, 0.75 expected"
    assert m.bias == torch.tensor([1.85], device=rank), f"{rank}: {m.bias.item()}, 1.85 expected"

    dist.destroy_process_group()


@skip_if_single_gpu
def test_step():
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(run_test_step, args=(world_size, temp_file_name), nprocs=world_size, join=True)


def run_test_step_with_closure(rank, world_size, tempfile_name, optimizer=None):
    dist_init(rank, world_size, tempfile_name)

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

    dist.destroy_process_group()


@skip_if_no_cuda
def test_step_with_closure():
    world_size = min(2, torch.cuda.device_count())
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(run_test_step_with_closure, args=(world_size, temp_file_name), nprocs=world_size, join=True)


def run_test_sharding(rank, world_size, tempfile_name):
    dist_init(rank, world_size, tempfile_name)
    params = []
    sizes = [9, 7, 5, 3]
    sizes_world = sizes * world_size

    for size in sizes_world:
        params.append(torch.rand(size, 1))

    # Make sure that the params are trainable, enforces size-based partitioning
    for p in params:
        p.requires_grad = True

    o = optim.OSS(params, lr=0.1)
    assert sum([x.numel() for x in o.optim.param_groups[0]["params"]]) == sum(sizes)

    dist.destroy_process_group()


def test_sharding():
    world_size = 4
    if torch.cuda.is_available():
        world_size = min(world_size, torch.cuda.device_count())

    _, temp_file_name = tempfile.mkstemp()
    mp.spawn(run_test_sharding, args=(world_size, temp_file_name), nprocs=world_size, join=True)


def run_test_collect_shards(rank, world_size, reference_rank, tempfile_name):
    dist_init(rank, world_size, tempfile_name)
    device = torch.device(rank) if torch.cuda.device_count() > 1 else DEVICE
    torch.cuda.set_device(rank)

    # Run a dummy step so that the optimizer state dict exists
    batch, input_width, hidden, target_width = 3, 3, 3, 5
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
        assert len(optimizer_state_dict["state"]) == len(list(model.parameters()))
    else:
        optimizer_state_dict = {}

    # distribute to the other ranks
    optimizer_state_dict = sync_object_ranks(optimizer_state_dict, reference_rank, device)

    # Load the optimizer state dict
    optimizer.load_state_dict(optimizer_state_dict)

    # Check that the states are not None, but {}
    for state in optimizer.state.values():
        for _, _ in state.items():
            pass

    # Test the state dict materialization on all ranks
    _ = optimizer.step(closure=closure)
    optimizer_state_dict = optimizer.state_dict(all_ranks=True)  # one per rank
    optimizer.load_state_dict(optimizer_state_dict)
    _ = optimizer.step(closure=closure)
    check_same_models_across_ranks(model, dist.group.WORLD, params_should_be_equal=True, check_broadcast_buffers=False)

    # Check that if the model is moved to cpu, the optimizer consolidation still works
    model.cpu()
    optimizer = optim.OSS(model.parameters(), lr=0.1, momentum=0.99)
    optimizer.consolidate_state_dict(recipient_rank=reference_rank)

    dist.destroy_process_group()


@skip_if_single_gpu
def test_collect_shards():
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]
    reference_rank = 0

    mp.spawn(
        run_test_collect_shards,
        args=(world_size, reference_rank, temp_file_name),
        nprocs=world_size,
        join=True,
    )


def run_test_reproducibility(rank, world_size, tempfile_name, broadcast_fp16):
    dist_init(rank, world_size, tempfile_name)
    device = torch.device(rank) if torch.cuda.device_count() > 1 else DEVICE
    torch.cuda.set_device(rank)

    # Run a dummy step so that the optimizer state dict exists
    batch, input_width, hidden, target_width = 3, 3, 3, 5
    target = torch.rand((batch, target_width), device=device)
    inputs = torch.rand((batch, input_width), device=device)

    model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width))
    model.to(device)
    model = DDP(model, device_ids=[device])

    loss_fn = torch.nn.L1Loss()
    loss_fn.to(device)

    optimizer = optim.OSS(model.parameters(), optim=torch.optim.RMSprop, lr=0.1, broadcast_fp16=broadcast_fp16)

    def closure():
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, target)
        loss.backward()
        return loss

    _ = optimizer.step(closure=closure)

    # Get a snapshot of the state at this point
    optimizer_state_dict = copy.deepcopy(optimizer.state_dict(all_ranks=True))
    model_state_dict = copy.deepcopy(model.state_dict())

    # Run two steps, log the loss
    _ = optimizer.step(closure=closure)
    reference_loss = optimizer.step(closure=closure)

    # Load the optimizer state dict, rewind the state two steps back
    optimizer.load_state_dict(optimizer_state_dict)
    model.load_state_dict(model_state_dict)

    # Run two new steps, log the loss again and check that we get the same
    _ = optimizer.step(closure=closure)
    test_loss = optimizer.step(closure=closure)

    assert torch.allclose(reference_loss, test_loss), f"{reference_loss} vs {test_loss}. Reproducibility is broken"

    # Check that no matter what the buffer is back to fp32
    for device in optimizer.buckets.keys():
        for bucket in optimizer.buckets[device].values():
            assert bucket.buffer.dtype == torch.float32
    dist.destroy_process_group()


@skip_if_single_gpu
@pytest.mark.parametrize("broadcast_fp16", [False, True])
def test_reproducibility(broadcast_fp16: bool):
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(
        run_test_reproducibility,
        args=(world_size, temp_file_name, broadcast_fp16),
        nprocs=world_size,
        join=True,
    )


def run_test_multiple_groups(rank, world_size, tempfile_name):
    # Only work with the even ranks, to check that the global_rank indexing is properly used
    dist_init(rank=rank, world_size=world_size, tempfile_name=tempfile_name, backend="gloo")
    sub_group_ranks = [0, 2, 4]
    process_group = torch.distributed.new_group(ranks=sub_group_ranks, backend="gloo")

    # Make sure that all the ranks get different training data
    # So that the sync check in between their models is meaningful
    torch.manual_seed(rank)
    np.random.seed(rank)

    # Standard deep learning setup
    device = "cpu"
    epochs, batch, input_width, hidden, target_width = 5, 3, 20, 10, 5
    loss_fn = torch.nn.L1Loss().to(device)

    def check(optimizer):
        # Just run a couple of epochs, check that the model is properly updated
        for _ in range(epochs):
            target = torch.rand((batch, target_width), device=device)
            inputs = torch.rand((batch, input_width), device=device)

            def closure():
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_fn(output, target)
                loss /= world_size
                loss.backward()
                dist.all_reduce(loss, group=process_group)  # Not strictly needed for the test below

                return loss

            _ = optimizer.step(closure=closure)

            # Check that all the params are the same on all ranks
            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    receptacle = [p.clone() for _ in sub_group_ranks] if rank == 0 else []
                    dist.gather(p, receptacle, dst=0, group=process_group)
                    if rank == 0:
                        for sync_p in receptacle[1:]:
                            assert torch.all(
                                torch.eq(receptacle[0], sync_p)
                            ), "Models differ in between ranks {} - {}".format(
                                torch.norm(receptacle[0]), torch.norm(sync_p)
                            )

    if rank in sub_group_ranks:
        # Model fitting in the broadcast bucket
        model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width)).to(
            device
        )

        # With SGD, Momentum is required to get a state to shard
        optimizer = optim.OSS(
            model.parameters(), lr=0.1, momentum=0.99, group=process_group, broadcast_buffer_size=2**20
        )
        check(optimizer)

        # Model not-fitting in the broadcast bucket
        model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width)).to(
            device
        )

        # With SGD, Momentum is required to get a state to shard
        optimizer = optim.OSS(model.parameters(), lr=0.1, momentum=0.99, group=process_group, broadcast_buffer_size=0)
        check(optimizer)

    dist.destroy_process_group(process_group)


@skip_if_py39_no_cuda
def test_multiple_groups():
    world_size = 6
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(
        run_test_multiple_groups,
        args=(world_size, temp_file_name),
        nprocs=world_size,
        join=True,
    )


def run_gradient_clipping(rank, world_size, tempfile_name):
    dist_init(rank, world_size, tempfile_name, backend="gloo")
    device = torch.device(rank)
    torch.manual_seed(rank)  # make sure that the different rank get different data

    # Run a dummy step so that the optimizer state dict exists
    batch, input_width, hidden, target_width = 3, 20, 10, 5
    target = torch.rand((batch, target_width), device=device)
    inputs = torch.rand((batch, input_width), device=device)
    NORMS = [1.0, 2.0, 1, 2, inf]
    CLIP_NORM = 0.3

    def check(norm):
        model_oss = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Linear(hidden, target_width),
        ).to(device)
        model = copy.deepcopy(model_oss)

        # For this test the gradients are (all) reduced in the same way in between the torch reference and fairscale.
        # Normally OSS would use ShardedDDP and only reduce to the proper rank, but this does not change the
        # gradient norm computation from OSS and adds a dependency.
        # to keep the comparison apples-to-apples DDP is used in both cases
        model_oss = DDP(
            module=model_oss,
            device_ids=[rank],
        )
        sharded_optimizer = optim.OSS(model_oss.parameters(), lr=0.1, momentum=0.99)

        model = DDP(
            model,
            device_ids=[rank],
        )

        loss_fn = torch.nn.L1Loss()
        loss_fn.to(device)

        model.zero_grad()
        model_oss.zero_grad()

        outputs = model(inputs)
        outputs_oss = model_oss(inputs)

        loss = loss_fn(outputs, target)
        loss.backward()

        loss_oss = loss_fn(outputs_oss, target)
        loss_oss.backward()
        torch.testing.assert_allclose(loss_oss, loss)

        # Check the equivalence with the non-sharded optim
        oss_total_norm = sharded_optimizer.clip_grad_norm(CLIP_NORM, norm_type=norm)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, norm_type=norm)
        assert torch.allclose(oss_total_norm, total_norm), "torch and fairscale should return the same grad norm"

        # Check that the params have indeed been clipped
        for params in sharded_optimizer._per_device_params.values():
            for param in filter(lambda x: x.grad is not None, params[rank]):
                assert torch.norm(param.grad, p=norm) < CLIP_NORM, f"param grad norm above clip : {param.grad}"

    for norm in NORMS:
        print(f"Checking norm {norm}")
        check(norm)

        # Check twice, catch an hypothetic iterator dumb mistake
        check(norm)

    dist.destroy_process_group()


@skip_if_no_cuda
def test_gradient_clipping():
    world_size = 3
    temp_file_name = tempfile.mkstemp()[1]

    if torch.cuda.is_available():
        world_size = min(world_size, torch.cuda.device_count())
    reference_rank = 0

    mp.spawn(
        run_gradient_clipping,
        args=(world_size, temp_file_name),
        nprocs=world_size,
        join=True,
    )


def run_state_dict_distributed(rank, world_size, tempfile_name):
    dist_init(rank, world_size, tempfile_name, backend="gloo")

    device = torch.device(rank)
    torch.manual_seed(rank)  # make sure that the different rank get different data

    # Setup two problems in parallel, we'll make sure that the second track (with save/load) follows the first one(untouched)
    # We split the model in two to test the multiple param groups support
    batch, input_width, hidden, target_width = 3, 20, 10, 5
    target = torch.rand((batch, target_width), device=device)
    inputs = torch.rand((batch, input_width), device=device)

    model_oss1 = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, hidden)).to(device)
    head_oss1 = torch.nn.Linear(hidden, target_width).to(device)

    model_oss2 = copy.deepcopy(model_oss1)
    head_oss2 = copy.deepcopy(head_oss1)

    # For this test the gradients are (all) reduced in the same way in between the torch reference and fairscale.
    # Normally OSS would use ShardedDDP and only reduce to the proper rank, but this does not change the
    # gradient norm computation from OSS and adds a dependency.
    # to keep the comparison apples-to-apples DDP is used in both cases
    model_oss1 = DDP(
        module=model_oss1,
        device_ids=[rank],
    )
    sharded_optimizer1 = optim.OSS(model_oss1.parameters(), lr=0.1, momentum=0.99)
    sharded_optimizer1.add_param_group({"params": head_oss1.parameters()})

    model_oss2 = DDP(
        module=model_oss2,
        device_ids=[rank],
    )
    sharded_optimizer2 = optim.OSS(model_oss2.parameters(), lr=0.1, momentum=0.99)
    sharded_optimizer2.add_param_group({"params": head_oss2.parameters()})

    loss_fn = torch.nn.L1Loss().to(device)

    def run_grad_step(model, head, optimizer):
        model.zero_grad()
        outputs = head(model(inputs))

    # pull the current state, broadcast it to all ranks
    sharded_optimizer2.consolidate_state_dict(recipient_rank=RECIPIENT_RANK)  # all ranks
    state_dict2 = sharded_optimizer2.state_dict() if rank == RECIPIENT_RANK else {}
    state_dict2 = sync_object_ranks(state_dict2, RECIPIENT_RANK, device)

    # re-create a new optimizer from scratch with absurd values, load the previous state
    sharded_optimizer2 = optim.OSS(model_oss2.parameters(), lr=1e6, momentum=0.0001)
    sharded_optimizer2.add_param_group({"params": head_oss2.parameters()})
    sharded_optimizer2.load_state_dict(state_dict2)
    check_same_model_params(
        model_oss1, model_oss2, "parameters of the two identical models have diverged (before any steps)"
    )

    # now take a step and check that parameters are equal
    run_grad_step(model_oss1, head_oss1, sharded_optimizer1)
    run_grad_step(model_oss2, head_oss2, sharded_optimizer2)
    check_same_model_params(
        model_oss1, model_oss2, "parameters of the two identical models have diverged (after stepping)"
    )

    # save the state dict for one model only, then distribute to the other ranks
    sharded_optimizer2.consolidate_state_dict(recipient_rank=RECIPIENT_RANK)  # all ranks
    state_dict2 = sharded_optimizer2.state_dict() if rank == RECIPIENT_RANK else {}
    state_dict2 = sync_object_ranks(state_dict2, RECIPIENT_RANK, device)

    # Check that the pulled state and the .param_groups attribute are in sync
    for replica in range(len(state_dict2["param_groups"])):
        for k in state_dict2["param_groups"][replica].keys():
            if k != "params":
                assert state_dict2["param_groups"][replica][k] == sharded_optimizer2.param_groups[0][k]

    # take a step
    run_grad_step(model_oss1, head_oss1, sharded_optimizer1)
    run_grad_step(model_oss2, head_oss2, sharded_optimizer2)
    check_same_model_params(
        model_oss1, model_oss2, "parameters of the two identical models have diverged (after consolidating)"
    )

    # save again for one rank, then distribute to the others
    sharded_optimizer2.consolidate_state_dict(recipient_rank=RECIPIENT_RANK)  # all ranks
    state_dict2 = sharded_optimizer2.state_dict() if rank == RECIPIENT_RANK else {}
    state_dict2 = sync_object_ranks(state_dict2, RECIPIENT_RANK, device)

    # reload the state_dict
    sharded_optimizer2 = optim.OSS(model_oss2.parameters(), lr=0.1, momentum=0.99)
    sharded_optimizer2.add_param_group({"params": head_oss2.parameters()})
    sharded_optimizer2.load_state_dict(state_dict2)

    # take a step
    run_grad_step(model_oss1, head_oss1, sharded_optimizer1)
    run_grad_step(model_oss2, head_oss2, sharded_optimizer2)
    check_same_model_params(
        model_oss1, model_oss2, "parameters of the two identical models have diverged (after reloading)"
    )

    dist.destroy_process_group()


@skip_if_single_gpu
def test_state_dict_distributed():
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    if torch.cuda.is_available():
        world_size = max(world_size, torch.cuda.device_count())

    mp.spawn(
        run_state_dict_distributed,
        args=(world_size, temp_file_name),
        nprocs=world_size,
        join=True,
    )


def run_ddp_parity(rank, world_size, backend, temp_file_name, change_train_graph, broadcast_fp16):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    np.random.seed(rank)
    hidden = 5
    in_channels = 3
    out_channels = 3
    batch = 64

    def check_optimizer_equivalence(optimizer: Type[torch.optim.Optimizer], change_train_graph: bool = False):
        # Any model works. Add one different buffer per rank
        trunk = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden), torch.nn.Linear(hidden, hidden), torch.nn.Linear(hidden, hidden)
        )
        trunk.register_buffer("test_buffer", torch.ones(1) * rank)
        trunk.to(device)

        head = torch.nn.Linear(hidden, out_channels).to(device)

        # Define a model to be trained by OSS
        oss_module = torch.nn.Sequential(trunk, head)

        # Make sure that the param groups are interleaved, to catch an ordering bug in the state dict
        oss_trainable_params = [
            {"params": list(trunk.parameters())[:-1] + list(head.parameters()), "lr": 1e-5},
            {"params": list(trunk.parameters())[-1], "lr": 1e-4},
        ]

        optimizer_settings: Dict[Any, Any] = {}
        if isinstance(optimizer, torch.optim.SGD):
            optimizer_settings["momentum"] = 0.9

        sharded_optimizer = optim.OSS(
            params=oss_trainable_params,
            optim=optimizer,
            group=None,
            broadcast_buffer_size=2**10,
            **optimizer_settings,
        )

        oss_ddp_model = DDP(module=oss_module, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)

        # Define a model to be trained by normal pytorch + DDP
        ddp_trunk = copy.deepcopy(trunk)
        ddp_head = copy.deepcopy(head)
        ddp_module = torch.nn.Sequential(ddp_trunk, ddp_head)

        ddp_trainable_params = [
            {"params": list(ddp_trunk.parameters())[:-1] + list(ddp_head.parameters()), "lr": 1e-5},
            {"params": list(ddp_trunk.parameters())[-1], "lr": 1e-4},
        ]
        ddp_optimizer = optimizer(ddp_trainable_params, **optimizer_settings)  # type: ignore
        ddp_model = DDP(module=ddp_module, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)

        def check_step():
            input_tensor = torch.rand((batch, in_channels)).to(device)

            def closure_ddp(input_tensor=input_tensor):
                ddp_optimizer.zero_grad()
                ddp_loss = ddp_model(input_tensor).abs().sum()
                ddp_loss.backward()
                return ddp_loss

            def closure_sharded(input_tensor=input_tensor):
                sharded_optimizer.zero_grad()
                sharded_loss = oss_ddp_model(input_tensor).abs().sum()
                sharded_loss.backward()
                return sharded_loss

            loss_ddp = cast(torch.Tensor, ddp_optimizer.step(closure=closure_ddp))
            loss_sharded_optim = cast(torch.Tensor, sharded_optimizer.step(closure=closure_sharded))

            assert torch.allclose(
                loss_ddp, loss_sharded_optim, rtol=1e-3
            ), f"Losses differ in between Pytorch optim and OSS\n {loss_ddp.item()} - {loss_sharded_optim.item()} - world size {world_size}"

            check_same_model_params(oss_ddp_model, ddp_model)

        # The model should be synchronized in between the ranks at construction time, check that
        check_same_model_params(oss_ddp_model, ddp_model)

        # The models should stay the same in between ddp and sharded optimizer
        for i in range(5):
            check_step()

            # Check that altering the trainable parameters does not cause DDP and OSS to diverge
            if change_train_graph:
                # Flip the first parameter from trainable to non-trainable and vice-versa
                next(ddp_module.parameters()).requires_grad = not next(ddp_module.parameters()).requires_grad
                next(oss_module.parameters()).requires_grad = not next(oss_module.parameters()).requires_grad
                # sharded_optimizer.refresh_trainable()

        # Check that the checkpoints are compatible (post pytorch 1.5)
        if torch_version()[1] > 5:
            # - get states
            ddp_state_dict = ddp_optimizer.state_dict()
            sharded_optimizer.consolidate_state_dict(recipient_rank=RECIPIENT_RANK)
            sharded_optim_state_dict = sharded_optimizer.state_dict() if rank == RECIPIENT_RANK else {}
            sharded_optim_state_dict = sync_object_ranks(sharded_optim_state_dict, RECIPIENT_RANK, device)

            # - cross load the states
            # run one step and check that the models are still the same
            ddp_state_dict_ref = copy.deepcopy(ddp_state_dict)  # OSS will remove some states
            ddp_optimizer.load_state_dict(sharded_optim_state_dict)  # mixup on purpose !
            sharded_optimizer.load_state_dict(ddp_state_dict)
            check_step()

            #  - self load, rewind, check no problem
            # run one step and check that the models are still the same
            ddp_optimizer.load_state_dict(ddp_state_dict_ref)
            sharded_optimizer.load_state_dict(sharded_optim_state_dict)
            check_step()

    for opt in [torch.optim.Adam, torch.optim.SGD]:
        check_optimizer_equivalence(opt, change_train_graph=change_train_graph)

    dist.destroy_process_group()


@pytest.mark.skip("broken at head")
@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("change_train_graph", [True, False])
@pytest.mark.parametrize("backend", [dist.Backend.NCCL, dist.Backend.GLOO])
@pytest.mark.parametrize("broadcast_fp16", [False, True])
def test_ddp_parity(change_train_graph: bool, backend: dist.Backend, broadcast_fp16: bool):
    temp_file_name = tempfile.mkstemp()[1]
    world_size = torch.cuda.device_count()
    mp.spawn(
        run_ddp_parity,
        args=(world_size, backend, temp_file_name, change_train_graph, broadcast_fp16),
        nprocs=world_size,
        join=True,
    )
