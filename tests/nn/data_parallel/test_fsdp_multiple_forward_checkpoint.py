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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from fairscale.nn import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import auto_wrap_bn
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, temp_files_ctx


class Model(nn.Module):
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


def create_model(with_fsdp, with_checkpoint):
    model = Model()
    if with_fsdp:
        # XXX: test auto_wrap_bn on & off.
        if True:
            model.block1 = auto_wrap_bn(model.block1, single_rank_pg=False)
            model.block2 = auto_wrap_bn(model.block2, single_rank_pg=False)
        if with_checkpoint:
            model.block2 = checkpoint_wrapper(model.block2, maintain_forward_counter=True)
        model.block1 = FSDP(model.block1)
        model.block2 = FSDP(model.block2)
        model.head = FSDP(model.head)
    else:
        if with_checkpoint:
            model.block2 = checkpoint_wrapper(model.block2, maintain_forward_counter=False)
    return model


def _distributed_worker(gpu_id, world_size, with_fsdp, with_checkpoint, files):
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

    model = create_model(with_fsdp, with_checkpoint)
    model = model.cuda()

    if with_fsdp:
        model = FSDP(model)
        context = contextlib.suppress()
        model.set_gradient_divide_factors(1.0, 2.0, True)
    else:
        # With DDP, we need no_sync and manual gradient reduction below because
        # it can't handle multiple forward pass + checkpointing otherwise.
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        context = model.no_sync()

    if gpu_id == 0:
        print(model)

    target = torch.LongTensor([0, 1, 2, 3, 4, 5]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    losses = {}
    i = 0
    with context:
        for iteration in range(3):
            out = model(batch)
            loss = criterion(out, target)
            print("Loss", iteration, ":", loss.item())
            losses[f"iter_{i}"] = loss.item()
            i += 1
            optimizer.zero_grad()
            loss.backward()
            if not with_fsdp:
                for p in model.parameters():
                    dist.all_reduce(p.grad.data)
                    p.grad.data.div_(2.0)
            optimizer.step()

    # Due to dist.all_reduce code block above with ddp.no_sync, we seem to hit a bug
    # in DDP where tensor.cpu() and torch.save() calls both hang. FSDP is not affected.
    # Therefore, we have to compare losses here instead of states.
    with open(filename_loss[rank], "wb") as f:
        pickle.dump(losses, f)

    teardown()


@skip_if_single_gpu
def test_multiple_forward_checkpoint():
    world_size = 2
    expected_losses = None
    # Ensure ddp == ddp+ckpt == fsdp == fsdp+ckpt.
    for with_fsdp in [False, True]:
        for with_checkpoint in [False, True]:
            # Get 4 files: 2 for dist_init and 2 for each rank to save the losses.
            with temp_files_ctx(num=2 + world_size) as temp_files:
                mp.spawn(_distributed_worker, (world_size, with_fsdp, with_checkpoint, temp_files), nprocs=world_size)
                final_losses = {}
                for rank in range(world_size):
                    with open(temp_files[2 + rank], "rb") as f:
                        final_losses[f"rank_{rank}"] = pickle.load(f)
                if expected_losses is None:
                    expected_losses = final_losses
                else:
                    print(f"fsdp: {with_fsdp} ckpt: {with_checkpoint}")
                    assert expected_losses == final_losses
