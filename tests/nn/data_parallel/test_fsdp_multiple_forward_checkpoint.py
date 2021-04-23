# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with multiple forward pass + checkpoint. """

import argparse
import contextlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from fairscale.nn import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import auto_wrap_bn


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
        """
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        """
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
        model.block1._id = "block1"
        model.block2 = FSDP(model.block2)
        model.block2._id = "block2"
        model.head = FSDP(model.head)
        model.head._id = "head"
    else:
        if with_checkpoint:
            model.block2 = checkpoint_wrapper(model.block2, maintain_forward_counter=False)
    return model


def _distributed_worker(gpu_id, with_fsdp, double_forward, with_checkpoint):
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:9099", world_size=2, rank=gpu_id)

    torch.manual_seed(0)
    # use False below to debug since error msg is not as good with cudnn.
    torch.backends.cudnn.enabled = True
    # these make things deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if double_forward:
        batch = [
            torch.randn(size=(2, 3, 224, 224)).cuda(),
            torch.randn(size=(2, 3, 96, 96)).cuda(),
            torch.randn(size=(2, 3, 96, 96)).cuda(),
        ]
    else:
        batch = torch.randn(size=(4, 3, 224, 224)).cuda()

    model = create_model(with_fsdp, with_checkpoint)
    model = model.cuda()

    if with_fsdp:
        model = FSDP(model)
        model._id = "root"
        context = contextlib.suppress()
        model.set_gradient_divide_factors(1.0, 2.0, True)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        context = model.no_sync()

    if gpu_id == 0:
        print(model)

    target = torch.LongTensor([0, 1, 2, 3, 4, 5]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    with context:
        for iteration in range(3):
            out = model(batch)
            loss = criterion(out, target)
            print("Loss", iteration, ":", loss.item())
            optimizer.zero_grad()
            loss.backward()
            if not with_fsdp:
                for p in model.parameters():
                    dist.all_reduce(p.grad.data)
                    p.grad.data.div_(2.0)
            optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fsdp", action="store_const", const=True, default=False)
    parser.add_argument("-d", "--double", action="store_const", const=True, default=False)
    parser.add_argument("-c", "--checkpoint", action="store_const", const=True, default=False)
    args = parser.parse_args()
    mp.spawn(_distributed_worker, (args.fsdp, args.double, args.checkpoint), nprocs=2)
