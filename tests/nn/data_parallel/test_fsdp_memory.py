# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with GPU memory usage. """

import gc

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
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, temp_files_ctx, torch_version


def get_global_group():
    """
    Singleton pytorch distributed group
    Inspired by https://github.com/pytorch/fairseq
    """
    if dist.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def to_fsdp(module):
    return FSDP(module, process_group=get_global_group())


def dump_all_tensors(rank):
    """Use this for debugging"""
    if rank != 0:
        return
    for obj in gc.get_objects():
        try:
            # Only need to check parameter type objects if asked.
            ttype = str(type(obj))
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                print(ttype, obj.shape, obj.dtype, obj.device, id(obj), obj.storage().size())
        except Exception as e:
            pass


def get_cur_mem(rank, result, prefix):
    """Collect memory allocated values in a result dict in MB"""
    result[prefix] = round(torch.cuda.memory_allocated() / 1024 / 1024)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def create_model(with_fsdp, with_checkpoint):
    model = Model()
    if with_fsdp:
        model.stem = auto_wrap_bn(model.stem, single_rank_pg=False)
        model.blocks = auto_wrap_bn(model.blocks, single_rank_pg=False)
        if with_checkpoint:
            model.blocks = checkpoint_wrapper(model.blocks)
        model.stem = to_fsdp(model.stem)
        model.blocks = to_fsdp(model.blocks)
        model.head = to_fsdp(model.head)
    else:
        if with_checkpoint:
            model.blocks = checkpoint_wrapper(model.blocks)
    return model


def _distributed_worker(gpu_id, world_size, with_fsdp, with_checkpoint, filename, filename_rpc, expected):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, filename, filename_rpc)
    assert result, "Dist init failed"

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    batch = torch.randn(size=(2, 3, 224, 224)).cuda()

    model = create_model(with_fsdp, with_checkpoint)
    model = model.cuda()
    if with_fsdp:
        model = to_fsdp(model)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id], bucket_cap_mb=500)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    results = {}
    for iteration in range(3):
        get_cur_mem(gpu_id, results, f"iter {iteration}: start")

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

    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize("ckpt", ["no_ckpt", "ckpt"])
@pytest.mark.parametrize("fsdp", ["ddp", "fsdp"])
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
    }[(fsdp, ckpt)]
    fsdp = fsdp == "fsdp"
    ckpt = ckpt == "ckpt"
    world_size = 2
    with temp_files_ctx(num=2) as temp_files:
        mp.spawn(
            _distributed_worker, (world_size, fsdp, ckpt, temp_files[0], temp_files[1], expected), nprocs=world_size
        )
