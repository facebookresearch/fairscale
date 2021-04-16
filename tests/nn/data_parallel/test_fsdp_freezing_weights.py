# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with some params frozen. """


import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.utils.testing import dist_init, objects_are_equal, rmf, skip_if_single_gpu, teardown


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.trunk(x))


def _create_model(with_fsdp: bool):
    model = Model()
    if with_fsdp:
        model.trunk = FSDP(model.trunk)
        model.head = FSDP(model.head)
    return model


def _distributed_worker(
    gpu_id: int, world_size: int, with_fsdp: bool, tempfile_name, unused, rank_0_output, expected_state
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    batch = torch.randn(size=(2, 3, 224, 224)).cuda()

    model = _create_model(with_fsdp)
    model = model.cuda()

    # freezing the trunk.
    for name, param in model.named_parameters():
        if "trunk" in name:
            param.requires_grad = False

    if with_fsdp:
        model = FSDP(model)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id])

    if gpu_id == 0:
        print(model)

    target = torch.LongTensor([0, 1]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for iteration in range(3):
        out = model(batch)
        fake_loss = criterion(out, target)
        print("Loss", iteration, ":", fake_loss.item())
        optimizer.zero_grad()
        fake_loss.backward()
        optimizer.step()

    if with_fsdp:
        fsdp_state = model.state_dict()
        # Move tensors to CPU to compare numerics.
        for k, v in fsdp_state.items():
            fsdp_state[k] = v.cpu()
        assert objects_are_equal(expected_state, fsdp_state, raise_exception=True)
    elif rank == 0:
        state_after = model.module.cpu().state_dict()
        torch.save(state_after, rank_0_output)

    teardown()


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    num = 6  # DDP and FSDP each needs 3 files.
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    yield tuple(files)

    # temp files could have been removed, so we use rmf.
    for name in files:
        rmf(name)


@skip_if_single_gpu
def tests1(temp_files):
    world_size = 2
    # DDP
    fsdp = False
    mp.spawn(_distributed_worker, (world_size, fsdp) + temp_files[:3] + (None,), nprocs=world_size)
    # FSDP
    fsdp = True
    expected_state = torch.load(temp_files[2])
    mp.spawn(_distributed_worker, (world_size, fsdp) + temp_files[3:] + (expected_state,), nprocs=world_size)
