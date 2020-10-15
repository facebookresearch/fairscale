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

from fairscale.nn.data_parallel import ShardedDataParallelExperimental

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")


def test_on_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"), world_size=4)


@skip_if_no_cuda
@skip_if_single_gpu
def test_on_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))


def run_one_step(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    # Any model works. Add one different buffer per rank
    model = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3)).to(device)
    model.register_buffer("test_buffer", torch.ones((1)) * rank)

    ddp = ShardedDataParallelExperimental(
        model_cpu=model,
        optimizer=torch.optim.SGD,
        optimizer_params={"lr": 0.1, "momentum": 0.99},
        world_size=world_size,
        device=device,
    )
    model = ddp.model
    optimizer = ddp.optimizer

    def check_same_model_params():
        # Check that all the params are the same on all ranks
        if dist.get_backend() != "nccl":
            for module in model.modules():
                for p in module.parameters():
                    # Check the params
                    gathered = [p.clone() for _ in range(world_size)] if rank == 0 else []
                    dist.gather(p, gathered, dst=0)
                    if rank == 0:
                        for sync_p in gathered[1:]:
                            assert torch.all(torch.eq(gathered[0], sync_p)), "Models differ in between ranks"

        # Check that all the buffers are in sync (authoritative rank is 0, its buffer is 0)
        for b in model.buffers():
            assert b.cpu().item() == 0.0

    # The model should be synchronized in between the ranks at construction time
    check_same_model_params()

    # Optim loop
    def closure():
        optimizer.zero_grad()

        input_tensor = torch.ones((64, 2)).to(device)
        loss = ddp(input_tensor).abs().sum()
        loss.backward()

        return loss

    # Run a couple of update loops
    for i in range(5):
        _ = optimizer.step(closure=closure)

    # Check that asking for a sync does sync the different ranks
    ddp.sync_ranks()
    check_same_model_params()

    dist.destroy_process_group()


def run_test(backend, device, world_size=2):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_one_step, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)
