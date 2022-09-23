# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing MultiProcessPipe Module
"""

import functools
import tempfile
from typing import Any, Dict, List, NamedTuple, Tuple

import pytest
import torch
import torch.distributed.autograd as dist_autograd
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn

from fairscale.experimental.nn.distributed_pipeline import DistributedLoss, DistributedPipeline, PipelineModulesGraph
from fairscale.fair_dev.testing.testing import skip_due_to_flakyness, skip_if_single_gpu
from fairscale.internal import torch_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch_version() < (1, 9, 0),
    reason="CPU tests fail right now and all tests require torch version >= 1.9.0.",
)

CPU_DEVICES = ["worker0/cpu", "worker1/cpu"]
GPU_DEVICES = ["worker0/cuda:0", "worker1/cuda:1"]
if torch.cuda.is_available():
    DEVICES = [CPU_DEVICES, GPU_DEVICES]
else:
    DEVICES = [CPU_DEVICES]


def rpc_worker(rank, world_size, init_file, func, *args):
    options = rpc.TensorPipeRpcBackendOptions(init_method="file://" + init_file)
    for i in range(world_size):
        options.set_device_map("worker" + str(i), {rank: i})
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


class RemoteModuleParams(NamedTuple):
    module_cls: nn.Module
    args: Tuple
    kwargs: Dict[str, Any]


def create_sequence_pipeline(
    layers: List[RemoteModuleParams], balance: List[int], devices: List[str], **kwargs: Any
) -> DistributedPipeline:
    """A simple helper function to create a pipeline from list of pipeline-modules that run sequentially.
    Args:
        layers: list of modules. They should not be already assigned a remote-device.
        balance: a list of integers how layers should be paritioned. Sum of numbers in 'balance'
            should be equal to the number of layers.
        devices: specification of remote device for each partition. Should be of the same length
            as 'balance'.
    """
    remote_modules: List[RemoteModule] = []
    index = 0
    for num_layers, remote_device in zip(balance, devices):
        next_index = index + num_layers
        for li in range(index, next_index):
            remote_modules.append(RemoteModule(remote_device, **layers[li]._asdict()))
        index = next_index

    graph = PipelineModulesGraph()
    graph.add_sequence(remote_modules, [0])

    return DistributedPipeline(graph, **kwargs)


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
    model = [RemoteModuleParams(nn.Linear, (4, 4), {})]
    pipe = create_sequence_pipeline(model, balance=[1], chunks=1, devices=devices[:1])


@rpc_test()
@skip_if_single_gpu
def create_multiple_layers():
    model = [RemoteModuleParams(nn.Linear, (4, 4), {}), RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1, 1], chunks=1, devices=["worker0/cpu", "worker0/cpu"])


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@skip_if_single_gpu
@skip_due_to_flakyness
def create_multiple_workers(devices):
    model = [RemoteModuleParams(nn.Linear, (4, 4), {}), RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1, 1], chunks=1, devices=devices[:2])


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@skip_if_single_gpu
def parameter_rrefs(devices):
    model = [RemoteModuleParams(nn.Linear, (4, 4), {}), RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1, 1], chunks=1, devices=devices[:2])
    parameter_rrefs = pipe.parameter_rrefs()
    assert len(parameter_rrefs) == 2


@rpc_test(world_size=1)
@pytest.mark.parametrize("devices", DEVICES)
def forward(devices):
    yh = torch.tensor([1.0, 0.0])
    x = torch.tensor([1.0, -1.0])
    model = [RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1], chunks=1, devices=devices[:1])
    y = pipe(x).to_here().cpu()
    assert torch.equal(y, yh), f"{y} != {yh}"


@rpc_test(world_size=1)
@pytest.mark.parametrize("devices", DEVICES)
def forward_chunks(devices):
    yh = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
    model = [RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1], chunks=4, devices=devices[:1])
    y = pipe(x).to_here().cpu()
    assert torch.equal(y, yh), f"{y} != {yh}"


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
@skip_if_single_gpu
def forward_multi(devices, checkpoint):
    device = devices[0].split("/")[1]
    torch.random.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    x = torch.randn(8, 4).to(device)
    x.requires_grad = True  # TODO(msb) remove this limitation
    model = [RemoteModuleParams(nn.Linear, (4, 4), {}), RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1, 1], chunks=4, devices=devices[:2], checkpoint=checkpoint)
    y = pipe(x).to_here()
    expected_sum = torch.tensor(5.0615)
    assert y.shape == torch.Size([8, 4])
    assert y.requires_grad is True
    assert torch.allclose(y.sum(), expected_sum), f"{y.sum()} != {expected_sum}"


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@skip_if_single_gpu
def backward(devices):
    device = devices[0].split("/")[1]
    torch.random.manual_seed(3)
    criterion = DistributedLoss(torch.nn.MSELoss)
    x = torch.randn(8, 4).to(device)
    model = [RemoteModuleParams(nn.Linear, (4, 4), {}), RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1, 1], chunks=4, devices=devices[:2])
    with dist_autograd.context() as context_id:
        y = pipe(x)
        loss = criterion(y, rpc.RRef(x))
        loss.backward(context_id)
        grads = dist_autograd.get_gradients(context_id)
    assert len(grads) == 2


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@skip_if_single_gpu
def update(devices):
    device = devices[0].split("/")[1]
    torch.random.manual_seed(3)
    criterion = DistributedLoss(torch.nn.MSELoss)
    x = torch.randn(8, 4).to(device)
    model = [RemoteModuleParams(nn.Linear, (4, 4), {}), RemoteModuleParams(nn.ReLU, (), {})]
    pipe = create_sequence_pipeline(model, balance=[1, 1], chunks=4, devices=devices[:2])
    opt = DistributedOptimizer(
        torch.optim.SGD,
        pipe.parameter_rrefs(),
        lr=0.05,
    )
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


class ConcatenateTensors(nn.Module):
    def forward(self, *inputs):
        return torch.cat(inputs, dim=1)


class SplitTensors(nn.Module):
    def forward(self, input):
        return torch.split(input, (input.shape[1] + 1) // 2, dim=1)


def extract_partitions(graph: PipelineModulesGraph, pipeline: DistributedPipeline) -> List[List[int]]:
    return [list(map(graph.nodes.index, p.nodes)) for p in pipeline.partitions]


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@skip_if_single_gpu
def multi_input_multi_output_layers(devices):
    device = devices[0].split("/")[1]
    torch.random.manual_seed(3)
    criterion = DistributedLoss(torch.nn.MSELoss)
    x = torch.randn(8, 4).to(device)

    #                                / ->linear_layer_2_1
    # input -> linear_layer1 -> split                     ->concatenate
    #                                \ ->linear_layer_2_2

    linear_layer_1 = RemoteModule(devices[0], nn.Linear, (4, 4), {})
    split = RemoteModule(devices[0], SplitTensors, (), {})
    linear_layers_2 = [
        RemoteModule(devices[0], nn.Linear, (2, 2), {}),
        RemoteModule(devices[1], nn.Linear, (2, 2), {}),
    ]
    concatenate = RemoteModule(devices[1], ConcatenateTensors, ())

    graph = PipelineModulesGraph()
    graph.add_sequence([linear_layer_1, split], [0], 2)
    for i, l in enumerate(linear_layers_2):
        graph.add_layer(l, [(split, i)])
    graph.add_layer(concatenate, linear_layers_2)

    pipe = DistributedPipeline(graph, chunks=4)
    assert [[0, 1], [2], [3], [4]] == extract_partitions(graph, pipe)
    parameter_rrefs = pipe.parameter_rrefs()
    assert len(parameter_rrefs) == 6
    opt = DistributedOptimizer(
        torch.optim.SGD,
        parameter_rrefs,
        lr=0.05,
    )
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


# A test for extracting the same graph as in test multi_input_multi_output_layers automatically
class ShardedLinearLayer(nn.Module):
    def __init__(self, input_device, shard_devices, output_device):
        super().__init__()
        self.split = RemoteModule(input_device, SplitTensors, (), {})
        self.linear_layers_2 = nn.ModuleList(
            [
                RemoteModule(shard_devices[0], nn.Linear, (2, 2), {}),
                RemoteModule(shard_devices[1], nn.Linear, (2, 2), {}),
            ]
        )
        self.concatenate = RemoteModule(output_device, ConcatenateTensors, ())

    def forward(self, input):
        shards = self.split(input)
        shards = [self.linear_layers_2[i](shards[i]) for i in range(2)]
        return self.concatenate(*shards)


@rpc_test(world_size=2)
@pytest.mark.parametrize("devices", DEVICES)
@skip_if_single_gpu
def auto_graph_extract(devices):
    from fairscale.experimental.nn.distributed_pipeline.trace import make_graph

    device = devices[0].split("/")[1]
    torch.random.manual_seed(3)
    criterion = DistributedLoss(torch.nn.MSELoss)
    x = torch.randn(8, 4).to(device)

    # create model
    model = nn.Sequential(
        RemoteModule(devices[0], nn.Linear, (4, 4), {}),
        ShardedLinearLayer(devices[0], devices, devices[1]),
        RemoteModule(devices[0], nn.Linear, (4, 4), {}),
    )
    graph = make_graph(model)
    pipe = DistributedPipeline(graph, chunks=4)
    partitions = extract_partitions(graph, pipe)
    assert [[0, 1], [2], [3], [4], [5]] == partitions, f"partitions={partitions}"
    parameter_rrefs = pipe.parameter_rrefs()
    assert len(parameter_rrefs) == 8
    opt = DistributedOptimizer(
        torch.optim.SGD,
        parameter_rrefs,
        lr=0.05,
    )
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
