# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing ShardedDDP
"""

from contextlib import suppress
import copy

import numpy as np
import pytest
import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.fair_dev.testing.testing import (
    check_same_model_params,
    skip_if_no_cuda,
    skip_if_single_gpu,
    temp_files_ctx,
)
from fairscale.internal import torch_version
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim import OSS

if torch_version() >= (1, 8, 0):
    from fairscale.optim.grad_scaler import ShardedGradScaler

"""
Check that ShardedDDP gets the same results as DDP in a variety of scenarii
"""

_test_fp16_reduction = [False]

if hasattr(dist, "algorithms.ddp_com_hooks.default_hooks"):
    _test_fp16_reduction.append(True)

_test_amp = [False]
if hasattr(torch.cuda.amp, "autocast"):
    _test_amp.append(True)

EMB_SIZE = 32
BATCH_SIZE = 8


def _get_mlp_emb(multiple_fw: bool = False):
    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3))
            self.head = Sequential(Linear(3, 3), Linear(3, 3))
            self.multiple_fw = multiple_fw
            self.embedding = torch.nn.Embedding(EMB_SIZE, 2)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:  # type: ignore
            inputs = self.embedding(indices)
            inputs = self.trunk(inputs)  # type: ignore

            if self.multiple_fw:
                return self.head(self.head(inputs))  # type: ignore

            return self.head(inputs)  # type: ignore

    return MLP()


def _get_random_inputs(device):
    return torch.floor(torch.rand((BATCH_SIZE, 2)) * EMB_SIZE).to(dtype=torch.long, device=device)


def run_ddp_parity(
    rank,
    world_size,
    backend,
    temp_file_name,
    reduce_buffer_size,
    grad_accumulation,
    change_train_graph,
    fp16_reduction,
    clip_grad_norm,
    amp,
    manual_reduction,
    multiple_fw,
):
    dist.init_process_group(init_method="file://" + temp_file_name, backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    np.random.seed(rank)
    NUMBER_BATCHS = 5

    # Test all combinations: AMP, Accumulate, Change train graph, reduce buckets
    print(
        f"{rank}: Checking configuration: accumulate {grad_accumulation}"
        + f" - change train graph {change_train_graph}"
        + f" - amp {amp}"
        + f" - manual reduction {manual_reduction}"
        + f" - buffers {reduce_buffer_size}"
        + f" - multiple FW {multiple_fw}",
        flush=True,
    )

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
    model = _get_mlp_emb(multiple_fw)
    model.register_buffer("test_buffer", torch.ones(1) * rank)
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
        reduce_fp16=fp16_reduction,
    )

    ddp_model_single = copy.deepcopy(model)
    ddp_optimizer = torch.optim.SGD(ddp_model_single.parameters(), lr=1e-4, momentum=0.99)
    ddp_model = DDP(ddp_model_single, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)

    if fp16_reduction:
        from dist.algorithms.ddp_com_hooks.default_hooks import fp16_compress_hook

        ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)  # type: ignore

    ddp_scaler = TorchGradScaler() if amp else None
    sharded_scaler = ShardedGradScaler() if amp else None

    # The model should be synchronized in between the ranks at construction time, check that
    check_same_model_params(sharded_ddp_model, ddp_model)

    # Typical training loop, check that we get the exact same results as DDP
    for i in range(NUMBER_BATCHS):
        input_tensor = _get_random_inputs(device)

        def ddp_closure(input_tensor=input_tensor):
            return closure(ddp_model, ddp_scaler, input_tensor, grad_accumulation)

        def sharded_closure(input_tensor=input_tensor):
            return closure(
                sharded_ddp_model,
                sharded_scaler,
                input_tensor,
                grad_accumulation,
                _manual_reduction=manual_reduction,
            )

        # Step/scale both
        for _scaler, _closure, _optimizer in (
            (ddp_scaler, ddp_closure, ddp_optimizer),
            (sharded_scaler, sharded_closure, sharded_optimizer),
        ):
            if _scaler is not None:
                _ = _closure(input_tensor)
                _scaler.step(_optimizer)
                _scaler.update()
            else:
                _optimizer.step(_closure())

        check_same_model_params(sharded_ddp_model, ddp_model, f"Rank: {rank} - Step {i} broke")

        # Check that the two grad norm are equivalent
        # NOTE: The grads can occasionally be NaNs, the scaler will skip the step in that case
        # This is not ShardedDDP specific. If the grads are not NaN for DDP then they should also
        # be valid for ShardedDDP
        # NOTE: DDP does not handle parameters trainability being changed after the fact, see
        # https://github.com/pytorch/pytorch/blob/5781aec74ef00284e0262817a649278c2e8072bf/torch/nn/parallel/distributed.py#L471
        if clip_grad_norm and not change_train_graph:
            if torch_version() >= (1, 9, 0):
                total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.3, norm_type=2.0, error_if_nonfinite=False)  # type: ignore
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.3, norm_type=2.0)  # type: ignore
            if not torch.isnan(total_norm):
                oss_total_norm = sharded_optimizer.clip_grad_norm(0.3, norm_type=2.0)
                allclose = torch.allclose(oss_total_norm, total_norm, atol=1e-2 if amp else 1e-8)

                if not allclose:
                    # Debug helper if this unit test does not pass, compare the gradients in between DDP and ShardedDDP
                    for idx, (p_ddp, p_sdp) in enumerate(zip(ddp_model.parameters(), sharded_ddp_model.parameters())):
                        if p_ddp.grad is not None:
                            if p_sdp.grad is not None:
                                print(rank, idx, torch.norm(p_ddp.grad), torch.norm(p_sdp.grad), flush=True)
                            else:
                                print(rank, idx, torch.norm(p_ddp.grad), "not owned", flush=True)

                assert (
                    allclose
                ), f"torch and fairscale should return the same grad norm\n {oss_total_norm} vs {total_norm}"
            else:
                print(rank, "NaN grad norm in DDP", flush=True)

        # Flip the trainability of the first parameter back and forth
        if i == 0 and change_train_graph:
            next(sharded_ddp_model.parameters()).requires_grad = not next(sharded_ddp_model.parameters()).requires_grad
            next(ddp_model.parameters()).requires_grad = not next(ddp_model.parameters()).requires_grad
            check_same_model_params(sharded_ddp_model, ddp_model, f"Rank: {rank} - Trainability refresh {i} broke")

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
@pytest.mark.parametrize("grad_accumulation", [True, False])
@pytest.mark.parametrize("change_train_graph", [True, False])
@pytest.mark.parametrize("fp16_reduction", _test_fp16_reduction)
@pytest.mark.parametrize("clip_grad_norm", [True, False])
@pytest.mark.parametrize("amp", _test_amp)
@pytest.mark.parametrize("manual_reduction", [True, False])
@pytest.mark.parametrize("multiple_fw", [True, False])
def test_ddp_parity(
    reduce_buffer_size,
    grad_accumulation,
    change_train_graph,
    fp16_reduction,
    clip_grad_norm,
    amp,
    manual_reduction,
    multiple_fw,
):
    if torch_version() < (1, 8, 0):
        pytest.skip("pytorch version >= 1.8.0 required")
    if manual_reduction and change_train_graph:
        pytest.skip("Skipping changing model and grad accumulation combination, makes little sense")

    world_size = torch.cuda.device_count()
    backend = dist.Backend.NCCL
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_ddp_parity,
            args=(
                world_size,
                backend,
                temp_files[0],
                reduce_buffer_size,
                grad_accumulation,
                change_train_graph,
                fp16_reduction,
                clip_grad_norm,
                amp,
                manual_reduction,
                multiple_fw,
            ),
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

    model = _get_mlp_emb()
    model.register_buffer("test_buffer", torch.ones(1) * rank)
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
        input_tensor = _get_random_inputs(device)

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
            sharded_ddp_model,
            ddp_model,
            f"DDP parity two optim test failing, step {i}, buffers {reduce_buffer_size}",
        )

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
@pytest.mark.parametrize("reduce_buffer_size", [0, 2**20])
def test_ddp_parity_two_optim(reduce_buffer_size):
    world_size = 2
    backend = dist.Backend.NCCL
    with temp_files_ctx(num=1) as temp_files:
        mp.spawn(
            run_ddp_parity_two_optim,
            args=(world_size, backend, temp_files[0], reduce_buffer_size),
            nprocs=world_size,
            join=True,
        )
