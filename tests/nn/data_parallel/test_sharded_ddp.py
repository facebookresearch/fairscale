# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing ShardedDDP
"""

from contextlib import suppress
import copy
import tempfile

import numpy as np
import pytest
import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.utils.testing import (
    GPT2,
    available_devices,
    check_same_model_params,
    check_same_models_across_ranks,
    skip_if_less_than_four_gpu,
    skip_if_no_cuda,
    skip_if_py38,
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
    rank, world_size, backend, device, temp_file_name, broadcast_buffers, grad_accumulation, reduce_buffer_size,
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

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-3, momentum=0.99)
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
        # when running on cpu/gloo the "nodes" are not really different
        same_params = device == torch.device("cpu") or grad_accumulation
        check_same_models_across_ranks(
            ddp_model, dist.group.WORLD, params_should_be_equal=same_params, check_broadcast_buffers=broadcast_buffers
        )

    dist.destroy_process_group()


def run_test(backend, device, world_size, broadcast_buffers, grad_accumulation, reduce_buffer_size):
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
def test_step_gpu(broadcast_buffers, grad_accumulation, reduce_buffer_size):
    world_size = 2
    run_test(
        dist.Backend.NCCL, torch.device("cuda"), world_size, broadcast_buffers, grad_accumulation, reduce_buffer_size
    )


@skip_if_py38
@pytest.mark.parametrize("broadcast_buffers", [True, False])
@pytest.mark.parametrize("grad_accumulation", [True, False])
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
def test_step_cpu(broadcast_buffers, grad_accumulation, reduce_buffer_size):
    world_size = 2
    run_test(
        dist.Backend.GLOO, torch.device("cpu"), world_size, broadcast_buffers, grad_accumulation, reduce_buffer_size
    )


def run_ddp_parity(
    rank, world_size, backend, temp_file_name, reduce_buffer_size, grad_accumulation, change_train_graph
):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    np.random.seed(rank)
    NUMBER_BATCHS = 5
    BATCH_SIZE = 8

    def check_parity(amp: bool, manual_reduction: bool):

        # The API should be the exact same in between the sharded and non-sharded variants, generic closure
        def closure(model, scaler, input_tensor, should_accumulate, _manual_reduction=False):
            accumulate_steps = 3 if should_accumulate else 1

            model.zero_grad()

            def step():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = model(input_tensor).abs().sum()
                        scaler.scale(loss).backward()
                else:
                    loss = model(input_tensor).abs().sum()
                    loss.backward()

            with model.no_sync() if should_accumulate else suppress():
                for _ in range(accumulate_steps - 1):
                    step()

            if not _manual_reduction:
                step()
            else:
                with model.no_sync():
                    step()

                model.reduce()

        # Any model works. Add one different buffer per rank
        model = _get_mlp()
        model.register_buffer("test_buffer", torch.ones((1)) * rank)
        model.to(device)

        # Make sure that the model starts with non-trainable, so that we check for the buckets to be
        # properly reassigned when/if this changes
        next(model.parameters()).requires_grad = False

        sharded_optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-4, momentum=0.99)
        sharded_ddp_model = ShardedDataParallel(
            module=model,
            sharded_optimizer=sharded_optimizer,
            broadcast_buffers=True,
            reduce_buffer_size=reduce_buffer_size,
        )

        ddp_model_single = copy.deepcopy(model)
        ddp_optimizer = torch.optim.SGD(ddp_model_single.parameters(), lr=1e-4, momentum=0.99)
        ddp_model = DDP(ddp_model_single, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)

        ddp_scaler = TorchGradScaler() if amp else None
        sharded_ddp_scaler = ShardedGradScaler() if amp else None

        # The model should be synchronized in between the ranks at construction time, check that
        check_same_model_params(sharded_ddp_model, ddp_model)

        # Typical training loop, check that we get the exact same results as DDP
        for i in range(NUMBER_BATCHS):
            input_tensor = torch.rand((BATCH_SIZE, 2)).to(device)

            def closure_ddp(input_tensor=input_tensor):
                return closure(ddp_model, ddp_scaler, input_tensor, grad_accumulation)

            def closure_sharded(input_tensor=input_tensor):
                return closure(
                    sharded_ddp_model,
                    sharded_ddp_scaler,
                    input_tensor,
                    grad_accumulation,
                    _manual_reduction=manual_reduction,
                )

            # Step/scale both
            if ddp_scaler is not None:
                _ = closure_ddp(input_tensor)
                ddp_scaler.step(ddp_optimizer)
                ddp_scaler.update()
            else:
                ddp_optimizer.step(closure=closure_ddp)

            if sharded_ddp_scaler is not None:
                _ = closure_sharded(input_tensor)
                sharded_ddp_scaler.step(sharded_optimizer)
                sharded_ddp_scaler.update()
            else:
                sharded_optimizer.step(closure=closure_sharded)

            check_same_model_params(sharded_ddp_model, ddp_model, f"Rank: {rank} - Step {i} broke")

            # Flip the trainability of the first parameter back and forth
            if i == 0 and change_train_graph:
                next(sharded_ddp_model.parameters()).requires_grad = not next(
                    sharded_ddp_model.parameters()
                ).requires_grad
                next(ddp_model.parameters()).requires_grad = not next(ddp_model.parameters()).requires_grad
                check_same_model_params(sharded_ddp_model, ddp_model, f"Rank: {rank} - Trainability refresh {i} broke")

    # Test all combinations: AMP, Accumulate, Change train graph, reduce buckets
    amp_tests = [False]
    if hasattr(torch.cuda.amp, "autocast"):
        amp_tests.append(True)

    manual_reductions = [False, True] if not grad_accumulation and not change_train_graph else [False]
    for manual_reduction in manual_reductions:
        for amp in amp_tests:
            print(
                f"Checking configuration: accumulate {grad_accumulation}"
                + f" - change train graph {change_train_graph}"
                + f" - amp {amp}"
                + f" - manual reduction {manual_reduction}"
                + f" - buffers {reduce_buffer_size}",
                flush=True,
            )
            check_parity(
                amp=amp, manual_reduction=manual_reduction,
            )

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
@pytest.mark.parametrize("grad_accumulation", [True, False])
@pytest.mark.parametrize("change_train_graph", [True, False])
def test_ddp_parity(reduce_buffer_size, grad_accumulation, change_train_graph):
    world_size = torch.cuda.device_count()
    backend = dist.Backend.NCCL
    mp.spawn(
        run_ddp_parity,
        args=(world_size, backend, tempfile.mkstemp()[1], reduce_buffer_size, grad_accumulation, change_train_graph),
        nprocs=world_size,
        join=True,
    )


def run_ddp_parity_two_optim(rank, world_size, backend, temp_file_name, reduce_buffer_size):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)
    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    np.random.seed(rank)  # Any model works. Add one different buffer per rank

    BATCHS = 20

    model = _get_mlp()
    model.register_buffer("test_buffer", torch.ones((1)) * rank)
    model.to(device)
    n_half_params = len(list(model.parameters())) // 2
    optim_settings = {"lr": 1e-3, "momentum": 0.99}

    sharded_optimizer = OSS(params=list(model.parameters())[:n_half_params], optim=torch.optim.SGD, **optim_settings)
    sharded_optimizer_2 = OSS(params=list(model.parameters())[n_half_params:], optim=torch.optim.SGD, **optim_settings)

    sharded_ddp_model = ShardedDataParallel(
        module=model,
        sharded_optimizer=[sharded_optimizer, sharded_optimizer_2],
        broadcast_buffers=True,
        reduce_buffer_size=reduce_buffer_size,
    )

    ddp_model_single = copy.deepcopy(model)
    ddp_optimizer = torch.optim.SGD(list(ddp_model_single.parameters())[:n_half_params], **optim_settings)
    ddp_optimizer_2 = torch.optim.SGD(list(ddp_model_single.parameters())[n_half_params:], **optim_settings)
    ddp_model = DDP(ddp_model_single, device_ids=[rank], broadcast_buffers=True)

    check_same_model_params(
        sharded_ddp_model,
        ddp_model,
        f"DDP parity two optim test failing. differing at startup, Buffers {reduce_buffer_size}",
    )

    for i in range(BATCHS):
        input_tensor = torch.rand((64, 2)).to(device)

        # Run DDP
        ddp_optimizer.zero_grad()
        ddp_optimizer_2.zero_grad()
        ddp_loss = ddp_model(input_tensor).abs().sum()
        ddp_loss.backward()
        ddp_optimizer.step()
        ddp_optimizer_2.step()
        torch.cuda.synchronize(device)

        # Run Sharded
        sharded_optimizer.zero_grad()
        sharded_optimizer_2.zero_grad()
        sharded_loss = sharded_ddp_model(input_tensor).abs().sum()
        sharded_loss.backward()
        sharded_optimizer.step()
        sharded_optimizer_2.step()
        torch.cuda.synchronize(device)

        check_same_model_params(
            sharded_ddp_model, ddp_model, f"DDP parity two optim test failing, step {i}, buffers {reduce_buffer_size}",
        )

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2 ** 20])
def test_ddp_parity_two_optim(reduce_buffer_size):
    world_size = 2
    backend = dist.Backend.NCCL
    mp.spawn(
        run_ddp_parity_two_optim,
        args=(world_size, backend, tempfile.mkstemp()[1], reduce_buffer_size),
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
    ddp_model = ShardedDataParallel(model, optimizer, reduce_buffer_size=0)

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
