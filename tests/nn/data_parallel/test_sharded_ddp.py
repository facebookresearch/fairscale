# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing OssDdp class.
"""

import tempfile
from typing import List

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential

from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim import OSS

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")
from contextlib import suppress

from fairscale.utils.testing import GPT2


def test_step_on_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"), world_size=4)


@skip_if_no_cuda
@skip_if_single_gpu
def test_step_on_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))


def run_one_step(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    def check(broadcast_buffers: bool, grad_accumulation: bool = False) -> None:
        # Any model works. Add one different buffer per rank
        model = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))
        model.register_buffer("test_buffer", torch.ones((1)) * rank)
        model.to(device)

        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
        ddp_model = ShardedDataParallel(model, optimizer, broadcast_buffers=broadcast_buffers)

        def check_same_model_params(same_params: bool):
            # Check that all the params are the same on all ranks
            # This should be true with and without broadcast_buffers, we don't have any real buffer here
            receptacle: List[torch.Tensor] = []

            if dist.get_backend() != "nccl":
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        # Check the params
                        receptacle = [p.clone() for _ in range(world_size)] if rank == 0 else []
                        dist.gather(p, receptacle, dst=0)
                        if rank == 0:
                            for sync_p in receptacle[1:]:
                                if same_params:
                                    assert torch.all(torch.eq(receptacle[0], sync_p)), "Models differ in between ranks"
                                else:
                                    assert not torch.all(
                                        torch.eq(receptacle[0], sync_p)
                                    ), "Gradients should not have been synced"

                # Check that all the buffers are in sync (authoritative rank is 0, its buffer is 0)
                if broadcast_buffers:
                    for b in ddp_model.buffers():
                        receptacle = [b.clone() for _ in range(world_size)] if rank == 0 else []
                        dist.gather(b, receptacle, dst=0)
                        if rank == 0:
                            for sync_b in receptacle[1:]:
                                if same_params:
                                    assert torch.all(torch.eq(receptacle[0], sync_b)), "Models differ in between ranks"
                                else:
                                    assert not torch.all(
                                        torch.eq(receptacle[0], sync_b)
                                    ), "Gradients should not have been synced"

                        assert b.cpu().item() == 0.0

        # The model should be synchronized in between the ranks at ShardedDataParallel construction time, check that
        check_same_model_params(same_params=True)

        # Optim loop
        def closure():
            optimizer.zero_grad()

            with ddp_model.no_sync() if grad_accumulation else suppress():
                input_tensor = torch.rand((64, 2)).to(device)
                loss = ddp_model(input_tensor).abs().sum()
                loss.backward()
            return loss

        # The models should stay the same in between the ranks
        for i in range(5):
            _ = optimizer.step(closure=closure)
            # when running on cpu/gloo the "nodes" are not really different
            same_params = device == torch.device("cpu") or grad_accumulation
            check_same_model_params(same_params=same_params)

    check(broadcast_buffers=False)
    check(broadcast_buffers=True)
    check(broadcast_buffers=False, grad_accumulation=True)
    check(broadcast_buffers=True, grad_accumulation=True)
    dist.destroy_process_group()


def run_test(backend, device, world_size=2):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_one_step, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_test_two_inputs(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    class _DoubleInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))

        def forward(self, x, y):
            x1 = self.mlp(x)
            x2 = self.mlp(y)
            return torch.cat((x1, x2), dim=1)

    model = _DoubleInput().to(device)

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    # Optim loop
    def closure():
        optimizer.zero_grad()
        input_tensor = torch.rand((64, 2)).to(device)
        loss = ddp_model(input_tensor, input_tensor).abs().sum()
        loss.backward()
        return loss

    for i in range(5):
        _ = optimizer.step(closure=closure)

    dist.destroy_process_group()


def test_inputs():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cpu"
    mp.spawn(run_test_two_inputs, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def test_ddp_attributes():
    # Check that ShardedDDP exposes the same attributes as Pytorch's DDP
    # - is multi_device_module
    # - device_type

    url = "file://" + tempfile.mkstemp()[1]
    dist.init_process_group(init_method=url, backend="gloo", rank=0, world_size=1)

    model = Sequential(Linear(2, 3), Linear(3, 3))
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert hasattr(ddp_model, "is_multi_device_module")
    assert hasattr(ddp_model, "device_type")
    dist.destroy_process_group()


def run_test_two_optimizers(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    class _DoubleInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))

        def forward(self, x, y):
            x1 = self.mlp(x)
            x2 = self.mlp(y)
            return torch.cat((x1, x2), dim=1)

    model = _DoubleInput().to(device)

    parameters = list(model.parameters())
    optimizer_1 = OSS(params=parameters[:-10], optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    optimizer_2 = OSS(params=parameters[-10:], optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, [optimizer_1, optimizer_2])

    # Optim loop
    def closure():
        input_tensor = torch.rand((64, 2)).to(device)
        loss = ddp_model(input_tensor, input_tensor).abs().sum()
        loss.backward()
        return loss

    for i in range(5):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        _ = optimizer_1.step(closure=closure)
        _ = optimizer_2.step(closure=closure)

    assert False

    dist.destroy_process_group()


def test_two_optimizers():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cpu"
    mp.spawn(run_test_two_optimizers, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_test_gpt2(rank, world_size, backend, device, temp_file_name):
    INPUT_DIM = 32
    BACH_SIZE = 10
    STEPS = 10

    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)
    model = GPT2(
        embed_dim=512, num_heads=2, num_layers=24, num_positions=INPUT_DIM * INPUT_DIM, num_vocab=512, num_classes=2
    ).to(device)
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    # Optim loop
    def closure():
        optimizer.zero_grad()
        # Force int inputs to prevent the first grad from firing
        input_tensor = torch.randint(10, (BACH_SIZE, INPUT_DIM)).to(device)
        loss = ddp_model(input_tensor).abs().sum()
        loss.backward()
        return loss

    # Check for bucketing overflows
    for i in range(STEPS):
        _ = optimizer.step(closure=closure)

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_gpt2():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(run_test_gpt2, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)
