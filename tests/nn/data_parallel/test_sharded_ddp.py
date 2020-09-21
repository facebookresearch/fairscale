# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing OssDdp class.
"""

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential

from fairscale.nn.data_parallel import ShardedDataParallel

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")


def test_on_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"))


@skip_if_no_cuda
@skip_if_single_gpu
def test_on_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))


def run_one_step(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    model = Sequential(Linear(2, 3), Linear(3, 4)).to(device)

    ddp = ShardedDataParallel(
        module=model, optimizer=torch.optim.SGD, optimizer_params={"lr": 0.1, "momentum": 0.99}, world_size=world_size
    )
    optimizer = ddp.optimizer

    input_tensor = torch.rand((64, 2)).to(device)
    output = ddp(input_tensor).abs().sum() / input_tensor.numel()
    output.backward()
    ddp.reduce()

    # Check that all the grads have been populated, for the shard
    if device == torch.device("cuda"):
        torch.cuda.synchronize()  # flush any remaining cuda op, just in case

    for pg in optimizer.optim.param_groups:
        for param in pg["params"]:
            if param.requires_grad:
                assert param.grad.abs().sum().item() > 0.0, "The reduce step should have populated all the gradients"

    # Check that the optimization process makes sense (ie. loss goes down for the same data)
    optimizer.step()
    new_eval = ddp(input_tensor).abs().sum() / input_tensor.numel()
    # assert new_eval.item() < output.item()


def run_test(backend, device, world_size=2):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_one_step, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_eval_mode(_unused):
    """ Testing eval mode make sure this is no asserts. """
    dist.init_process_group(
        init_method=f"file://{tempfile.mkstemp()[1]}", backend=dist.Backend.GLOO, rank=0, world_size=1
    )
    model = Sequential(Linear(2, 3), Linear(3, 4))
    optimizer_params = {"lr": 0.1, "momentum": 0.99}
    ddp = ShardedDataParallel(model, torch.optim.SGD, optimizer_params, 1)
    optimizer = ddp.optimizer

    ddp.eval()
    for _ in range(5):
        input_tensor = torch.rand((64, 2))
        output = ddp(input_tensor)

    ddp.train()
    try:
        for _ in range(5):
            input_tensor = torch.rand((64, 2))
            output = ddp(input_tensor)
    except RuntimeError:
        pass
    else:
        assert False, "Multiple forward passes on training mode should not pass"


def test_eval_mode():
    mp.spawn(run_eval_mode, args=(), join=True)
