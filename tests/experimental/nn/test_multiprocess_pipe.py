# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing MultiProcessPipe Module
"""

import functools
import tempfile

import pytest
import torch
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn

from fairscale.experimental.nn.multiprocess_pipe import DistributedLoss, MultiProcessPipe
from fairscale.utils.testing import torch_version

BOUNCE_TENSORS = True

CPU_DEVICES = ["worker0/cpu", "worker1/cpu"]
GPU_DEVICES = ["worker0/cuda:0", "worker1/cuda:1"]
if torch.cuda.is_available():
    DEVICES = [CPU_DEVICES, GPU_DEVICES]
else:
    DEVICES = [CPU_DEVICES]

# cuda test is because of https://github.com/pytorch/pytorch/issues/54266
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch_version() < (1, 8, 0), reason="requires torch version >= 1.8.0 and cuda"
)


def rpc_worker(rank, world_size, init_file, func, *args):
    # Workaround for https://github.com/pytorch/pytorch/issues/54266
    if not torch.cuda.is_available():
        options = rpc.ProcessGroupRpcBackendOptions(init_method="file://" + init_file)
        rpc.init_rpc(
            "worker" + str(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.PROCESS_GROUP,
            rpc_backend_options=options,
        )
    else:
        # Workaround for https://github.com/pytorch/pytorch/issues/53844
        if torch_version() == (1, 8, 0):
            options = rpc.TensorPipeRpcBackendOptions(init_method="file://" + init_file, _transports=["ibv", "uv"])
        else:
            options = rpc.TensorPipeRpcBackendOptions(init_method="file://" + init_file)
        rpc.init_rpc(
            "worker" + str(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
    if rank == 0:
        func(*args)
    rpc.shutdown()


def rpc_test(world_size=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mp.spawn(rpc_worker, args=(world_size, tempfile.mkstemp()[1], func, *kwargs.values()), nprocs=world_size)

        globals()["test_" + func.__name__] = wrapper
        return func

    return decorator


@rpc_test()
@pytest.mark.parametrize("devices", DEVICES)
def create(devices):
    model = [("linear", nn.Linear, (4, 4), {})]
    pipe = MultiProcessPipe(model, balance=[1], chunks=1, devices=devices[:1])


@rpc_test()
def create_multiple_layers():
    model = [("linear1", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1, 1], chunks=1, devices=["worker0/cpu", "worker0/cpu"])


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
def create_multiple_workers(devices):
    model = [("linear1", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1, 1], chunks=1, devices=devices[:2])


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
def parameter_rrefs(devices):
    model = [("linear1", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1, 1], chunks=1, devices=devices[:2])
    parameter_rrefs = pipe.parameter_rrefs()
    assert len(parameter_rrefs) == 2


@rpc_test(world_size=1)
@pytest.mark.parametrize("devices", DEVICES)
def forward(devices):
    yh = torch.tensor([1.0, 0.0])
    x = torch.tensor([1.0, -1.0])
    model = [("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1], chunks=1, devices=devices[:1])
    y = pipe(x).to_here().cpu()
    assert torch.equal(y, yh), f"{y} != {yh}"


@rpc_test(world_size=1)
@pytest.mark.parametrize("devices", DEVICES)
def forward_chunks(devices):
    yh = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
    model = [("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1], chunks=4, devices=devices[:1])
    y = pipe(x).to_here().cpu()
    assert torch.equal(y, yh), f"{y} != {yh}"


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
def forward_multi(devices):
    torch.random.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    x = torch.randn(8, 4)
    model = [("linear1", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1, 1], chunks=4, devices=devices[:2])
    if BOUNCE_TENSORS:
        y = pipe(x).remote().cpu().to_here()
    else:
        y = pipe(x).to_here()
    expected_sum = torch.tensor(5.0615)
    assert y.shape == torch.Size([8, 4])
    assert y.requires_grad is True
    assert torch.allclose(y.sum(), expected_sum), f"{y.sum()} != {expected_sum}"


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
def backward(devices):
    torch.random.manual_seed(3)
    criterion = DistributedLoss(torch.nn.MSELoss)
    x = torch.randn(8, 4)
    model = [("linear1", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1, 1], chunks=4, devices=devices[:2])
    with dist_autograd.context() as context_id:
        y = pipe(x)
        loss = criterion(y, rpc.RRef(x))
        loss.backward(context_id)
        grads = dist_autograd.get_gradients(context_id)
    assert len(grads) == 2


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
def update(devices):
    torch.random.manual_seed(3)
    criterion = DistributedLoss(torch.nn.MSELoss)
    x = torch.randn(8, 4)
    model = [("linear1", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
    pipe = MultiProcessPipe(model, balance=[1, 1], chunks=4, devices=devices[:2])
    params = pipe.parameter_rrefs()
    opt = DistributedOptimizer(torch.optim.SGD, pipe.parameter_rrefs(), lr=0.05,)
    losses = []
    for i in range(2):
        with dist_autograd.context() as context_id:
            y = pipe(x)
            loss = criterion(y, rpc.RRef(x))
            losses.append(loss)
            loss.backward(context_id)
            opt.step(context_id)
    losses = [l.to_here() for l in losses]
    assert losses[0] > losses[1], f"{losses[0]} !> {losses[1]}"
