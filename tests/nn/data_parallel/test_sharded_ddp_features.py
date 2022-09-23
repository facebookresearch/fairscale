# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing ShardedDDP
"""

from contextlib import suppress

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential

from fairscale.fair_dev.testing.testing import (
    GPT2,
    SGDWithPausingCompute,
    available_devices,
    check_same_models_across_ranks,
    skip_if_less_than_four_gpu,
    skip_if_no_cuda,
    skip_if_single_gpu,
    temp_files_ctx,
)
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim import OSS


def _get_mlp(tripwire: bool = False):
    if not tripwire:
        return Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))

    class Tripwire(torch.nn.Module):
        """A model made to expose possible corner cases"""

        def __init__(self) -> None:
            super().__init__()
            self.model = Linear(2, 3, bias=False)

            # mismatched types in between trainable or not, can trip the buckets for instance
            self.register_parameter("tripwire", torch.nn.Parameter(torch.LongTensor((3, 3)), requires_grad=False))

        def forward(self, x):
            return self.model(x)

    return Tripwire()


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
    reduce_fp16=False,
):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    # Any model works. Add one different buffer per rank
    model = _get_mlp()
    model.register_buffer("test_buffer", torch.ones(1) * rank)
    model.to(device)

    next(model.parameters()).requires_grad = False  # Test non-trainable parameters

    optimizer_settings = {"lr": 1e-3, "momentum": 0.99}
    if optimizer_type == SGDWithPausingCompute:
        optimizer_settings["rank"] = rank

    optimizer = OSS(params=model.parameters(), optim=optimizer_type, **optimizer_settings)
    ddp_model = ShardedDataParallel(
        model,
        optimizer,
        broadcast_buffers=broadcast_buffers,
        reduce_buffer_size=reduce_buffer_size,
        reduce_fp16=reduce_fp16,
    )

    # The model should be synchronized in between the ranks at ShardedDataParallel construction time, check that
    check_same_models_across_ranks(
        ddp_model, dist.group.WORLD, params_should_be_equal=True, check_broadcast_buffers=broadcast_buffers
    )

    # Optim loop
    def closure():
        ddp_model.zero_grad(set_to_none=True)

        with ddp_model.no_sync() if grad_accumulation else suppress():
            input_tensor = torch.rand((64, 2)).to(device)
            loss = ddp_model(input_tensor).abs().sum()

            # If grad_accumulation, we can check after the forward that the models are different
            # (not synced)
            if grad_accumulation:
                check_same_models_across_ranks(
                    ddp_model, dist.group.WORLD, params_should_be_equal=False, check_broadcast_buffers=True
                )

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
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_one_step,
            args=(world_size, backend, device, temp_files[0], broadcast_buffers, grad_accumulation, reduce_buffer_size),
            nprocs=world_size,
            join=True,
        )


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("broadcast_buffers", [True, False])
@pytest.mark.parametrize("grad_accumulation", [True, False])
@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, SGDWithPausingCompute])
@pytest.mark.parametrize("reduce_fp16", [False, True])
@pytest.mark.parametrize(
    "setup",
    [
        [dist.Backend.NCCL, torch.device("cuda")],
        [dist.Backend.GLOO, torch.device("cpu")],
        [dist.Backend.GLOO, torch.device("cuda")],
    ],
)
def test_step(broadcast_buffers, grad_accumulation, reduce_buffer_size, optimizer_type, reduce_fp16, setup):
    world_size = 2
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_one_step,
            args=(
                world_size,
                setup[0],
                setup[1],
                temp_files[0],
                broadcast_buffers,
                grad_accumulation,
                reduce_buffer_size,
                optimizer_type,
                reduce_fp16,
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
        ddp_model.zero_grad(set_to_none=True)
        input_tensor = torch.rand((64, 2)).to(device)
        loss = ddp_model(input_tensor, input_tensor).abs().sum()
        loss.backward()
        return loss

    for _ in range(5):
        _ = optimizer.step(closure=closure)

    dist.destroy_process_group()


@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("device", available_devices)
@skip_if_single_gpu
def test_inputs(reduce_buffer_size, backend, device):
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    if backend == "nccl" and device == "cpu":
        pytest.skip("Incompatible combination, or cuda not available")
        return
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_two_inputs,
            args=(world_size, backend, device, temp_files[0], reduce_buffer_size),
            nprocs=world_size,
            join=True,
        )


def test_ddp_attributes():
    # Check that ShardedDDP exposes the same attributes as Pytorch's DDP
    # - is multi_device_module
    # - device_type
    with temp_files_ctx(num=1) as temp_files:
        dist.init_process_group(init_method="file://" + temp_files[0], backend="gloo", rank=0, world_size=1)

        model = Sequential(Linear(2, 3), Linear(3, 3))
        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
        ddp_model = ShardedDataParallel(model, optimizer)

        assert hasattr(ddp_model, "is_multi_device_module")
        assert hasattr(ddp_model, "device_type")
        assert hasattr(ddp_model, "module")
        dist.destroy_process_group()


def test_random_attributes():
    with temp_files_ctx(num=1) as temp_files:
        # Check that ShardedDDP exposes the original module's attributes
        dist.init_process_group(init_method="file://" + temp_files[0], backend="gloo", rank=0, world_size=1)

        model = Sequential(Linear(2, 3), Linear(3, 3))
        model.banana = "sweet"

        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
        ddp_model = ShardedDataParallel(model, optimizer)

        assert hasattr(ddp_model, "banana")
        assert not hasattr(ddp_model, "orange")

        dist.destroy_process_group()


def test_catch_grad_grad():
    with temp_files_ctx(num=1) as temp_files:
        # Check that ShardedDDP exposes the original module's attributes
        dist.init_process_group(init_method="file://" + temp_files[0], backend="gloo", rank=0, world_size=1)

        model = Sequential(Linear(2, 3), Linear(3, 3))
        model.train()
        chained_grad = torch.zeros_like(next(model.parameters()))
        chained_grad.requires_grad = True
        next(model.parameters()).grad = chained_grad

        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
        ddp_model = ShardedDataParallel(model, optimizer)

        inputs = torch.rand(100, 2)
        with pytest.raises(RuntimeError):
            _ = ddp_model(inputs)

        dist.destroy_process_group()


def test_mixed_types():
    with temp_files_ctx(num=1) as temp_files:
        # Check that ShardedDDP exposes the original module's attributes
        dist.init_process_group(init_method="file://" + temp_files[0], backend="gloo", rank=0, world_size=1)

        model = _get_mlp(tripwire=True)

        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
        model = ShardedDataParallel(model, optimizer)
        input_tensor = torch.rand((2, 2))
        _ = model(input_tensor)

        dist.destroy_process_group()


def run_test_train_eval_change(rank, world_size, file):
    # Check that ShardedDDP handles the switch from training to eval properly
    dist.init_process_group(init_method="file://" + file, backend="gloo", rank=rank, world_size=world_size)

    model = _get_mlp()
    model.train()
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    model = ShardedDataParallel(model, optimizer)
    input_tensor = torch.rand((2, 2))
    loss = model(input_tensor).sum()
    loss.backward()  # make sure that the gradients are reduced

    # Wipe the gradients and switch to eval mode
    model.zero_grad()
    model.eval()
    _ = model(input_tensor)
    assert next(model.parameters()).grad is None or torch.norm(next(model.parameters()).grad) < 1e-6

    # Get back to training
    model = model.train()
    model(input_tensor).sum().backward()
    assert torch.norm(next(model.parameters()).grad) > 0.0

    dist.destroy_process_group()


def test_train_eval_change():
    world_size = 4
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_train_eval_change,
            args=(world_size, temp_files[0]),
            nprocs=world_size,
            join=True,
        )


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

    # Check that we can change the data type
    ddp_model.to(device=torch.device("cpu"), dtype=torch.float16)

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
def test_device_change(reduce_buffer_size):
    # Check that ShardedDDP handles a device change properly
    world_size = 2
    backend = "nccl"
    with temp_files_ctx(num=1) as temp_files:
        device = "cuda"
        mp.spawn(
            run_test_device_change,
            args=(world_size, backend, device, temp_files[0], reduce_buffer_size),
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
@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
def test_training_change(reduce_buffer_size):
    world_size = 2
    backend = "nccl"
    device = "cuda"
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_training_change,
            args=(world_size, backend, device, temp_files[0], reduce_buffer_size),
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
    device = "cuda"
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_ddp_sync_batch_norm,
            args=(world_size, backend, device, temp_files[0]),
            nprocs=world_size,
            join=True,
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
    device = "cpu"
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_two_optimizers, args=(world_size, backend, device, temp_files[0]), nprocs=world_size, join=True
        )


def run_test_gpt2(rank, world_size, backend, device, temp_file_name, reduce_buffer_size):
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
    )
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer, reduce_buffer_size=reduce_buffer_size)

    # Move the model to another device post-construction
    model = model.to(device)

    # Optim loop
    set_to_none = True

    def closure():
        nonlocal set_to_none
        ddp_model.zero_grad(set_to_none=set_to_none)
        set_to_none = not set_to_none

        # Force int inputs to prevent the first grad from firing
        input_tensor = torch.randint(10, (BACH_SIZE, INPUT_DIM)).to(device)
        loss = ddp_model(input_tensor).abs().sum()
        loss.backward()
        return loss

    # Check for bucketing overflows
    for i in range(STEPS):
        _ = optimizer.step(closure=closure)

        # Stress test the .to() method
        ddp_model.to(device=device, dtype=torch.float16)
        ddp_model.to(device=device, dtype=torch.float32)

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("reduce_buffer", [2**23, 2**40])
def test_gpt2(world_size, reduce_buffer):
    # Check that having trainable unused params is fine
    backend = "gloo"
    device = "cuda"
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_gpt2,
            args=(world_size, backend, device, temp_files[0], reduce_buffer),
            nprocs=world_size,
            join=True,
        )


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
@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_multiple_groups(reduce_buffer_size, backend):
    world_size = 4
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_test_multiple_groups,
            args=(world_size, temp_files[0], backend, reduce_buffer_size),
            nprocs=world_size,
            join=True,
        )
