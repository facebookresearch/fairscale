# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing ShardedDDP
"""

from contextlib import suppress
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential

from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim import OSS
from fairscale.utils.testing import (
    GPT2,
    SGDWithPausingCompute,
    available_devices,
    check_same_models_across_ranks,
    skip_if_less_than_four_gpu,
    skip_if_no_cuda,
    skip_if_single_gpu,
)


def _get_mlp():
    return Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))


class _DoubleInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _get_mlp()

    def forward(self, x, y):
        x1 = self.mlp(x)
        x2 = self.mlp(y)
        return torch.cat((x1, x2), dim=1)


def run_one_step(
    rank,
    world_size,
    backend,
    device,
    temp_file_name,
    broadcast_buffers,
    grad_accumulation,
    reduce_buffer_size,
    optimizer_type,
):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    # Any model works. Add one different buffer per rank
    model = _get_mlp()
    model.register_buffer("test_buffer", torch.ones((1)) * rank)
    model.to(device)

    next(model.parameters()).requires_grad = False  # Test non-trainable parameters

    optimizer_settings = {"lr": 1e-3, "momentum": 0.99}
    if optimizer_type == SGDWithPausingCompute:
        optimizer_settings["rank"] = rank

    optimizer = OSS(params=model.parameters(), optim=optimizer_type, **optimizer_settings)
    ddp_model = ShardedDataParallel(
        model, optimizer, broadcast_buffers=broadcast_buffers, reduce_buffer_size=reduce_buffer_size
    )

    # The model should be synchronized in between the ranks at ShardedDataParallel construction time, check that
    check_same_models_across_ranks(
        ddp_model, dist.group.WORLD, params_should_be_equal=True, check_broadcast_buffers=broadcast_buffers
    )

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

        # For a sync of all the streams
        if device.type == torch.device("cuda").type:
            torch.cuda.synchronize(device=device)

        # when running on cpu/gloo the "nodes" are not really different
        same_params = device == torch.device("cpu") or not grad_accumulation
        check_same_models_across_ranks(
            ddp_model, dist.group.WORLD, params_should_be_equal=same_params, check_broadcast_buffers=broadcast_buffers
        )

    dist.destroy_process_group()


def run_test(backend, device, world_size, broadcast_buffers, grad_accumulation, reduce_buffer_size, optimizer_type):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(
        run_one_step,
        args=(world_size, backend, device, temp_file_name, broadcast_buffers, grad_accumulation, reduce_buffer_size),
        nprocs=world_size,
        join=True,
    )


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("broadcast_buffers", [True, False])
@pytest.mark.parametrize("grad_accumulation", [True, False])
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, SGDWithPausingCompute])
@pytest.mark.parametrize(
    "setup",
    [
        [dist.Backend.NCCL, torch.device("cuda")],
        [dist.Backend.GLOO, torch.device("cpu")],
        [dist.Backend.GLOO, torch.device("cuda")],
    ],
)
def test_step(broadcast_buffers, grad_accumulation, reduce_buffer_size, optimizer_type, setup):
    world_size = 2
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(
        run_one_step,
        args=(
            world_size,
            setup[0],
            setup[1],
            temp_file_name,
            broadcast_buffers,
            grad_accumulation,
            reduce_buffer_size,
            optimizer_type,
        ),
        nprocs=world_size,
        join=True,
    )


def run_test_two_inputs(rank, world_size, backend, device, temp_file_name, reduce_buffer_size):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    model = _DoubleInput().to(device)
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer, reduce_buffer_size=reduce_buffer_size)

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


@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("device", available_devices)
def test_inputs(reduce_buffer_size, backend, device):
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    if backend == "nccl" and device == "cpu":
        pytest.skip("Incompatible combination, or cuda not available")
        return

    mp.spawn(
        run_test_two_inputs,
        args=(world_size, backend, device, tempfile.mkstemp()[1], reduce_buffer_size),
        nprocs=world_size,
        join=True,
    )


def test_ddp_attributes():
    # Check that ShardedDDP exposes the same attributes as Pytorch's DDP
    # - is multi_device_module
    # - device_type
    dist.init_process_group(init_method="file://" + tempfile.mkstemp()[1], backend="gloo", rank=0, world_size=1)

    model = Sequential(Linear(2, 3), Linear(3, 3))
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert hasattr(ddp_model, "is_multi_device_module")
    assert hasattr(ddp_model, "device_type")
    dist.destroy_process_group()


def test_random_attributes():
    # Check that ShardedDDP exposes the original module's attributes
    dist.init_process_group(init_method="file://" + tempfile.mkstemp()[1], backend="gloo", rank=0, world_size=1)

    model = Sequential(Linear(2, 3), Linear(3, 3))
    model.banana = "sweet"

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert hasattr(ddp_model, "banana")
    assert not hasattr(ddp_model, "orange")

    dist.destroy_process_group()


def run_test_device_change(rank, world_size, backend, device, temp_file_name, reduce_buffer_size):
    # Check that the wrapped module can change devices
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = Sequential(Linear(2, 3), Linear(3, 3)).cpu()  # not device on purpose, test changing it after the fact
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(
        model, optimizer, sync_models_at_startup=False, reduce_buffer_size=reduce_buffer_size
    )
    try:
        ddp_model.to(device)
        assert False, "Changing devices should be caught and not supported"
    except AssertionError:
        pass

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
def test_device_change(reduce_buffer_size):
    # Check that ShardedDDP is compatible with sync batch norm across multiple GPUs
    world_size = 2
    backend = "nccl"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(
        run_test_device_change,
        args=(world_size, backend, device, temp_file_name, reduce_buffer_size),
        nprocs=world_size,
        join=True,
    )


def run_test_training_change(rank, world_size, backend, device, temp_file_name, reduce_buffer_size):
    group = dist.init_process_group(
        init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)

    model = Sequential(Linear(2, 3), Linear(3, 3)).to(device)
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer, process_group=group, reduce_buffer_size=reduce_buffer_size)

    inputs = torch.rand((10, 2), device=device)
    outputs = ddp_model(inputs)  # assert if the module has not been changed properly
    _ = outputs.norm().backward()

    ddp_model.eval()
    ddp_model(inputs)  # This will assert if eval() is not properly taken into account
    ddp_model(inputs)

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
def test_training_change(reduce_buffer_size):
    world_size = 2
    backend = "nccl"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(
        run_test_training_change,
        args=(world_size, backend, device, temp_file_name, reduce_buffer_size),
        nprocs=world_size,
        join=True,
    )


def run_test_ddp_sync_batch_norm(rank, world_size, backend, device, temp_file_name):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)

    model = Sequential(Linear(2, 3), torch.nn.BatchNorm1d(3), Linear(3, 3)).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)  # in pytorch 1.5 syncBN switches to the default device/cpu

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert isinstance(model[1], torch.nn.SyncBatchNorm)
    # Ensures sync batch norm handles have been added
    ddp_model(torch.randn(2, 2).to(device))
    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_ddp_sync_batch_norm():
    # Check that ShardedDDP is compatible with sync batch norm across multiple GPUs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(
        run_test_ddp_sync_batch_norm, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True
    )


def run_test_two_optimizers(rank, world_size, backend, device, temp_file_name):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)
    model = _DoubleInput().to(device)

    parameters = list(model.parameters())
    optimizer_1 = OSS(params=parameters[:-10], optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    optimizer_2 = OSS(params=parameters[-10:], optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
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

    dist.destroy_process_group()


def test_two_optimizers():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cpu"
    mp.spawn(run_test_two_optimizers, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_test_gpt2(rank, world_size, backend, device, temp_file_name):
    INPUT_DIM = 16
    BACH_SIZE = 10
    STEPS = 10

    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)
    model = GPT2(
        embed_dim=256, num_heads=2, num_layers=12, num_positions=INPUT_DIM * INPUT_DIM, num_vocab=512, num_classes=2
    ).to(device)
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
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


def run_test_multiple_groups(rank, world_size, tempfile_name, backend, reduce_buffer_size):
    # Only work with the even ranks, to check that the global_rank indexing is properly used
    dist.init_process_group(init_method="file://" + tempfile_name, backend=backend, rank=rank, world_size=world_size)

    sub_group_ranks = [0, 2]
    process_group = torch.distributed.new_group(ranks=sub_group_ranks, backend=backend)

    # Make sure that all the ranks get different training data
    # So that the sync check in between their models is meaningful
    torch.manual_seed(rank)
    np.random.seed(rank)

    # Standard deep learning setup
    device = "cuda"
    torch.cuda.set_device(rank)

    epochs, batch, input_width, hidden, target_width = 5, 3, 20, 10, 5
    loss_fn = torch.nn.L1Loss().to(device)

    def check(optimizer, model):
        # Just run a couple of epochs, check that the model is properly updated
        for _ in range(epochs):
            target = torch.rand((batch, target_width), device=device)
            inputs = torch.rand((batch, input_width), device=device)

            def closure():
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_fn(output, target)
                loss.backward()
                return loss

            _ = optimizer.step(closure=closure)

            # Check that all the params are the same on all ranks
            check_same_models_across_ranks(
                model, process_group, params_should_be_equal=True, check_broadcast_buffers=True
            )

    if rank in sub_group_ranks:
        # Model not-fitting in the broadcast bucket
        model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width)).to(
            device
        )

        # With SGD, Momentum is required to get a state to shard
        optimizer = OSS(model.parameters(), group=process_group, lr=1e-3, momentum=0.99)
        model = ShardedDataParallel(
            model, optimizer, process_group=process_group, reduce_buffer_size=reduce_buffer_size
        )
        check(optimizer, model)

    dist.destroy_process_group(process_group)


@skip_if_less_than_four_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_multiple_groups(reduce_buffer_size, backend):
    world_size = 4
    temp_file_name = tempfile.mkstemp()[1]

    mp.spawn(
        run_test_multiple_groups,
        args=(world_size, temp_file_name, backend, reduce_buffer_size),
        nprocs=world_size,
        join=True,
    )
