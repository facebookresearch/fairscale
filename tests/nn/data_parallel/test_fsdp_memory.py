# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with GPU memory usage. """

import contextlib

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from fairscale.nn import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import auto_wrap_bn
from fairscale.utils.parallel import get_global_group
from fairscale.utils.testing import (
    dist_init,
    dump_all_tensors,
    skip_if_single_gpu,
    teardown,
    temp_files_ctx,
    torch_version,
)


def to_fsdp(module, fsdp_config):
    return FSDP(module, process_group=get_global_group(), **fsdp_config)


def get_cur_mem(rank, result, prefix):
    """Collect memory allocated values in a result dict in MB"""
    result[prefix] = round(torch.cuda.memory_allocated() / 1024 / 1024)


class Model(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # TODO (Min): for both fast and memory efficient conv kernels, we should be using
        #     AMP/fp16 + channel_last input format. Otherwise, cudnn internally does conversion
        #     to channel_last when it is fp16 weights. Leave this knowledge here and perhaps
        #     future test can cover it.
        self.stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def create_model(with_fsdp, with_checkpoint, model_hidden_dim, fsdp_config):
    model = Model(model_hidden_dim)
    if with_fsdp:
        model.stem = auto_wrap_bn(model.stem, single_rank_pg=False)
        model.blocks = auto_wrap_bn(model.blocks, single_rank_pg=False)
        if with_checkpoint:
            model.blocks = checkpoint_wrapper(model.blocks)
        model.stem = to_fsdp(model.stem, fsdp_config)
        model.blocks = to_fsdp(model.blocks, fsdp_config)
        model.head = to_fsdp(model.head, fsdp_config)
    else:
        if with_checkpoint:
            model.blocks = checkpoint_wrapper(model.blocks)
    return model


def _distributed_worker(
    gpu_id, world_size, with_fsdp, with_checkpoint, filename, filename_rpc, expected, model_hidden_dim, fsdp_config
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, filename, filename_rpc)
    assert result, "Dist init failed"

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Note that FSDP auto-cast the input in AMP mode. So we don't need to call half() here.
    batch = torch.randn(size=(2, 3, 224, 224)).cuda()

    model = create_model(with_fsdp, with_checkpoint, model_hidden_dim, fsdp_config)
    model = model.cuda()
    if with_fsdp:
        model = to_fsdp(model, fsdp_config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id], bucket_cap_mb=500)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    # Set AMP context if needed.
    context = contextlib.suppress()
    if "mixed_precision" in fsdp_config and fsdp_config["mixed_precision"]:
        context = torch.cuda.amp.autocast(enabled=True)

    results = {}  # results of memory stats
    for iteration in range(3):
        get_cur_mem(gpu_id, results, f"iter {iteration}: start")

        with context:
            out = model(batch)
            get_cur_mem(gpu_id, results, f"iter {iteration}: after fwd")

            out = sum(o.sum() for o in out[0])
            fake_loss = criterion(out, torch.tensor(0.0).cuda())
            get_cur_mem(gpu_id, results, f"iter {iteration}: after loss")

        fake_loss.backward()
        get_cur_mem(gpu_id, results, f"iter {iteration}: after bwd")

        optimizer.step()
        get_cur_mem(gpu_id, results, f"iter {iteration}: after step")

        # It is important to use `set_to_none` below, not optimizer.zero_grad() to reclaim memory.
        if torch_version() >= (1, 7, 0):
            model.zero_grad(set_to_none=True)
        else:
            for p in model.parameters():
                p.grad = None
        get_cur_mem(gpu_id, results, f"iter {iteration}: done")

    assert results == expected, f"{results} but expected {expected}"
    dump_all_tensors(gpu_id)

    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize("ckpt", ["no_ckpt", "ckpt"])
@pytest.mark.parametrize("fsdp", ["ddp", "fsdp", "fsdp_amp_default", "fsdp_amp_compute_dtype32"])
def test_fsdp_memory(fsdp, ckpt):
    expected = {
        ("ddp", "no_ckpt"): {
            "iter 0: start": 9,
            "iter 0: after fwd": 346,
            "iter 0: after loss": 346,
            "iter 0: after bwd": 14,
            "iter 0: after step": 14,
            "iter 0: done": 9,
            "iter 1: start": 9,
            "iter 1: after fwd": 346,
            "iter 1: after loss": 346,
            "iter 1: after bwd": 14,
            "iter 1: after step": 14,
            "iter 1: done": 9,
            "iter 2: start": 9,
            "iter 2: after fwd": 346,
            "iter 2: after loss": 346,
            "iter 2: after bwd": 14,
            "iter 2: after step": 14,
            "iter 2: done": 9,
        },
        ("fsdp", "no_ckpt"): {
            "iter 0: start": 3,
            "iter 0: after fwd": 340,
            "iter 0: after loss": 340,
            "iter 0: after bwd": 66,
            "iter 0: after step": 66,
            "iter 0: done": 3,
            "iter 1: start": 3,
            "iter 1: after fwd": 340,
            "iter 1: after loss": 340,
            "iter 1: after bwd": 66,
            "iter 1: after step": 66,
            "iter 1: done": 3,
            "iter 2: start": 3,
            "iter 2: after fwd": 340,
            "iter 2: after loss": 340,
            "iter 2: after bwd": 66,
            "iter 2: after step": 66,
            "iter 2: done": 3,
        },
        ("fsdp_amp_default", "no_ckpt"): {
            "iter 0: start": 105,
            "iter 0: after fwd": 1285,
            "iter 0: after loss": 1285,
            "iter 0: after bwd": 259,
            "iter 0: after step": 259,
            "iter 0: done": 105,
            "iter 1: start": 105,
            "iter 1: after fwd": 1285,
            "iter 1: after loss": 1285,
            "iter 1: after bwd": 259,
            "iter 1: after step": 259,
            "iter 1: done": 105,
            "iter 2: start": 105,
            "iter 2: after fwd": 1285,
            "iter 2: after loss": 1285,
            "iter 2: after bwd": 259,
            "iter 2: after step": 259,
            "iter 2: done": 105,
        },
        ("fsdp_amp_compute_dtype32", "no_ckpt"): {
            "iter 0: start": 105,
            "iter 0: after fwd": 1388,
            "iter 0: after loss": 1388,
            "iter 0: after bwd": 272,
            "iter 0: after step": 272,
            "iter 0: done": 105,
            "iter 1: start": 105,
            "iter 1: after fwd": 1388,
            "iter 1: after loss": 1388,
            "iter 1: after bwd": 272,
            "iter 1: after step": 272,
            "iter 1: done": 105,
            "iter 2: start": 105,
            "iter 2: after fwd": 1388,
            "iter 2: after loss": 1388,
            "iter 2: after bwd": 272,
            "iter 2: after step": 272,
            "iter 2: done": 105,
        },
        ("ddp", "ckpt"): {
            "iter 0: start": 9,
            "iter 0: after fwd": 57,
            "iter 0: after loss": 57,
            "iter 0: after bwd": 14,
            "iter 0: after step": 14,
            "iter 0: done": 9,
            "iter 1: start": 9,
            "iter 1: after fwd": 57,
            "iter 1: after loss": 57,
            "iter 1: after bwd": 14,
            "iter 1: after step": 14,
            "iter 1: done": 9,
            "iter 2: start": 9,
            "iter 2: after fwd": 57,
            "iter 2: after loss": 57,
            "iter 2: after bwd": 14,
            "iter 2: after step": 14,
            "iter 2: done": 9,
        },
        ("fsdp", "ckpt"): {
            "iter 0: start": 3,
            "iter 0: after fwd": 51,
            "iter 0: after loss": 51,
            "iter 0: after bwd": 66,
            "iter 0: after step": 66,
            "iter 0: done": 3,
            "iter 1: start": 3,
            "iter 1: after fwd": 51,
            "iter 1: after loss": 51,
            "iter 1: after bwd": 66,
            "iter 1: after step": 66,
            "iter 1: done": 3,
            "iter 2: start": 3,
            "iter 2: after fwd": 51,
            "iter 2: after loss": 51,
            "iter 2: after bwd": 66,
            "iter 2: after step": 66,
            "iter 2: done": 3,
        },
        ("fsdp_amp_default", "ckpt"): {
            "iter 0: start": 105,
            "iter 0: after fwd": 130,
            "iter 0: after loss": 130,
            "iter 0: after bwd": 259,
            "iter 0: after step": 259,
            "iter 0: done": 105,
            "iter 1: start": 105,
            "iter 1: after fwd": 130,
            "iter 1: after loss": 130,
            "iter 1: after bwd": 259,
            "iter 1: after step": 259,
            "iter 1: done": 105,
            "iter 2: start": 105,
            "iter 2: after fwd": 130,
            "iter 2: after loss": 130,
            "iter 2: after bwd": 259,
            "iter 2: after step": 259,
            "iter 2: done": 105,
        },
        ("fsdp_amp_compute_dtype32", "ckpt"): {
            "iter 0: start": 105,
            "iter 0: after fwd": 130,
            "iter 0: after loss": 130,
            "iter 0: after bwd": 272,
            "iter 0: after step": 272,
            "iter 0: done": 105,
            "iter 1: start": 105,
            "iter 1: after fwd": 130,
            "iter 1: after loss": 130,
            "iter 1: after bwd": 272,
            "iter 1: after step": 272,
            "iter 1: done": 105,
            "iter 2: start": 105,
            "iter 2: after fwd": 130,
            "iter 2: after loss": 130,
            "iter 2: after bwd": 272,
            "iter 2: after step": 272,
            "iter 2: done": 105,
        },
    }[(fsdp, ckpt)]

    # Compute the FSDP config.
    fsdp_config = {}

    # Set mixed precision.
    if "amp" in fsdp:
        fsdp_config["mixed_precision"] = True

    # When compute_dtype is FP32, make sure we use clear_autocast_cache.
    # Setting fp32_reduce_scatter and verbose for more code coverage.
    if "compute_dtype32" in fsdp:
        fsdp_config["compute_dtype"] = torch.float32
        fsdp_config["fp32_reduce_scatter"] = True
        fsdp_config["clear_autocast_cache"] = True
        fsdp_config["verbose"] = True

    # Using bigger hidden dimension for AMP to increase the model size
    # so that bug in handling params will show up but we don't do that
    # in the base case to keep the test fast.
    #   hidden_dim 128: model size 4MB
    #   hidden_dim 1024: model size 200MB
    model_hidden_dim = 128
    if "amp" in fsdp:
        model_hidden_dim = 1024

    # Get the fsdp and checkpoint flags.
    with_fsdp = "fsdp" in fsdp
    with_ckpt = ckpt == "ckpt"

    world_size = 2
    with temp_files_ctx(num=2) as temp_files:
        mp.spawn(
            _distributed_worker,
            (world_size, with_fsdp, with_ckpt, temp_files[0], temp_files[1], expected, model_hidden_dim, fsdp_config),
            nprocs=world_size,
        )
