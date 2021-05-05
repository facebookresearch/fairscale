# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with multiple forward pass + checkpoint. """

import contextlib
import pickle

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from fairscale.nn import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import auto_wrap_bn
from fairscale.nn.wrap import enable_wrap, wrap
from fairscale.utils.testing import (
    dist_init,
    objects_are_equal,
    skip_if_single_gpu,
    teardown,
    temp_files_ctx,
    torch_version,
)


class Model(nn.Module):
    """Model to test FSDP(checkpoint())."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.head(self.block2(self.block1(x)))
        elif isinstance(x, list):
            ys = [self.head(self.block2(self.block1(e))) for e in x]
            return torch.cat(ys, dim=0)


class Model2(nn.Module):
    """Model to test FSDP(checkpoint(), checkpoint())."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),)
        self.block2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=False),)
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(inplace=True),)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(128, 10))

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.head(self.block3(self.block2(self.block1(x))))
        elif isinstance(x, list):
            ys = [self.head(self.block3(self.block2(self.block1(e)))) for e in x]
            return torch.cat(ys, dim=0)


def create_model(with_model2, with_fsdp, with_checkpoint, mixed_precision, flatten, wrap_bn, fp32_reduce_scatter):
    model = Model2() if with_model2 else Model()
    if with_fsdp:
        if wrap_bn:
            model.block1 = auto_wrap_bn(model.block1, single_rank_pg=False)
            model.block2 = auto_wrap_bn(model.block2, single_rank_pg=False)
            if with_model2:
                model.block3 = auto_wrap_bn(model.block3, single_rank_pg=False)
        if with_checkpoint:
            model.block2 = checkpoint_wrapper(model.block2, maintain_forward_counter=True)
            if with_model2:
                model.block3 = checkpoint_wrapper(model.block3, maintain_forward_counter=True)
        with enable_wrap(
            wrapper_cls=FSDP,
            flatten_parameters=flatten,
            mixed_precision=mixed_precision,
            compute_dtype=torch.float32,
            fp32_reduce_scatter=fp32_reduce_scatter,
        ):
            model.block1 = wrap(model.block1)
            model.block2 = wrap(model.block2)
            if with_model2:
                model.block3 = wrap(model.block3)
            model.head = wrap(model.head)
    else:
        if with_checkpoint:
            model.block2 = checkpoint_wrapper(model.block2, maintain_forward_counter=False)
            if with_model2:
                model.block3 = checkpoint_wrapper(model.block3, maintain_forward_counter=False)
    return model


def _distributed_worker(
    gpu_id,
    world_size,
    with_model2,
    with_fsdp,
    with_checkpoint,
    files,
    mixed_precision,
    flatten,
    wrap_bn,
    fp32_reduce_scatter,
):
    filename, filename_rpc = files[:2]
    filename_loss = files[2:]

    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, filename, filename_rpc)
    assert result, "Dist init failed"

    # use False below to debug since error msg is not as good with cudnn.
    torch.backends.cudnn.enabled = True

    # these make things deterministic.
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure we have multiple forward passes.
    batch = [
        torch.randn(size=(2, 3, 224, 224)).cuda(),
        torch.randn(size=(2, 3, 96, 96)).cuda(),
        torch.randn(size=(2, 3, 96, 96)).cuda(),
    ]

    if mixed_precision and not with_fsdp:
        batch = [x.half() for x in batch]

    model = create_model(
        with_model2, with_fsdp, with_checkpoint, mixed_precision, flatten, wrap_bn, fp32_reduce_scatter
    )
    model = model.cuda()

    if with_fsdp:
        model = FSDP(
            model,
            flatten_parameters=flatten,
            mixed_precision=mixed_precision,
            compute_dtype=torch.float32,
            fp32_reduce_scatter=fp32_reduce_scatter,
        )
        model.set_gradient_divide_factors(1.0, 2.0, True)
        no_sync_context = contextlib.suppress()
    else:
        # With DDP, we need no_sync and manual gradient reduction below because
        # it can't handle multiple forward pass + checkpointing otherwise.
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        no_sync_context = model.no_sync()

    mp_context = contextlib.suppress()
    if mixed_precision:
        mp_context = torch.cuda.amp.autocast(enabled=True)

    if gpu_id == 0:
        print(model)

    target = torch.LongTensor([0, 1, 2, 3, 4, 5]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    losses = {}
    i = 0
    with no_sync_context:
        for iteration in range(3):
            with mp_context:
                out = model(batch)
                loss = criterion(out, target)
                print("Loss", iteration, ":", loss.item())
                losses[f"iter_{i}"] = loss
                i += 1
                optimizer.zero_grad()
                loss.backward()
            # Manual grad reduction, no autocast.
            if not with_fsdp:
                for p in model.parameters():
                    dist.all_reduce(p.grad.data)
                    p.grad.data.div_(2.0)
            # Stepping, no autocast
            optimizer.step()

    # Due to dist.all_reduce code block above with ddp.no_sync, we seem to hit a bug
    # in DDP where tensor.cpu() and torch.save() calls both hang. FSDP is not affected.
    # Therefore, we have to compare losses here instead of states.
    with open(filename_loss[rank], "wb") as f:
        pickle.dump(losses, f)

    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
@pytest.mark.parametrize("wrap_bn", ["auto_wrap_bn", "no_auto_wrap_bn"])
@pytest.mark.parametrize("model_type", ["model1", "model2"])
def test_multiple_forward_checkpoint(precision, flatten, wrap_bn):
    mixed_precision = precision == "mixed"
    flatten = flatten == "flatten"
    wrap_bn = wrap_bn == "auto_wrap_bn"
    fp32_reduce_scatter = True if mixed_precision else None
    with_model2 = model_type == "model2"

    if torch_version() < (1, 8, 0) and flatten:
        # 1.6 and 1.7 throws this error:
        #   RuntimeError: Trying to backward through the graph a second time, but the saved
        #   intermediate results have already been freed. Specify retain_graph=True when calling
        #   backward the first time.
        pytest.skip("older pytorch throws error when flatten is used")

    world_size = 2
    expected_losses = None
    # Ensure ddp == ddp+ckpt == fsdp == fsdp+ckpt.
    for with_fsdp in [False, True]:
        for with_checkpoint in [False, True]:
            # Get 4 files: 2 for dist_init and 2 for each rank to save the losses.
            with temp_files_ctx(num=2 + world_size) as temp_files:
                mp.spawn(
                    _distributed_worker,
                    (
                        world_size,
                        with_model2,
                        with_fsdp,
                        with_checkpoint,
                        temp_files,
                        mixed_precision,
                        flatten,
                        wrap_bn,
                        fp32_reduce_scatter,
                    ),
                    nprocs=world_size,
                )
                final_losses = {}
                for rank in range(world_size):
                    with open(temp_files[2 + rank], "rb") as f:
                        final_losses[f"rank_{rank}"] = pickle.load(f)
                if expected_losses is None:
                    expected_losses = final_losses
                else:
                    print(f"fsdp: {with_fsdp} ckpt: {with_checkpoint}")
                    assert objects_are_equal(expected_losses, final_losses, raise_exception=True)
