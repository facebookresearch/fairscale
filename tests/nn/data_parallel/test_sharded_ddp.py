# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing OssDdp class.
"""

import tempfile

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

    def check(broadcast_buffers: bool) -> None:
        # Any model works. Add one different buffer per rank
        model = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))
        model.register_buffer("test_buffer", torch.ones((1)) * rank)
        model.to(device)

        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
        ddp_model = ShardedDataParallel(model, optimizer, broadcast_buffers=broadcast_buffers)

        def check_same_model_params():
            # Check that all the params are the same on all ranks
            # This should be true with and without broadcast_buffers, we don't have any real buffer here
            if dist.get_backend() != "nccl":
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        # Check the params
                        receptacle = [p.clone() for _ in range(world_size)] if rank == 0 else []
                        dist.gather(p, receptacle, dst=0)
                        if rank == 0:
                            for sync_p in receptacle[1:]:
                                assert torch.all(torch.eq(receptacle[0], sync_p)), "Models differ in between ranks"

                # Check that all the buffers are in sync (authoritative rank is 0, its buffer is 0)
                if broadcast_buffers:
                    for b in ddp_model.buffers():
                        receptacle = [b.clone() for _ in range(world_size)] if rank == 0 else []
                        dist.gather(b, receptacle, dst=0)
                        if rank == 0:
                            for sync_b in receptacle[1:]:
                                assert torch.all(torch.eq(receptacle[0], sync_b)), "Models differ in between ranks"
                        assert b.cpu().item() == 0.0

        # The model should be synchronized in between the ranks at ShardedDataParallel construction time, check that
        check_same_model_params()

        # Optim loop
        def closure():
            optimizer.zero_grad()

            input_tensor = torch.rand((64, 2)).to(device)
            loss = ddp_model(input_tensor).abs().sum()
            loss.backward()
            return loss

        # The models should stay the same in between the ranks
        for i in range(5):
            _ = optimizer.step(closure=closure)
            check_same_model_params()

    check(broadcast_buffers=False)
    check(broadcast_buffers=True)

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

    # The models should stay the same in between the ranks
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
