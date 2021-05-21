# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from fairscale.nn import FullyShardedDataParallel
from fairscale.utils.testing import in_temporary_directory, skip_if_single_gpu, temp_files_ctx


def _check_split_worker(gpu_id: int, sync_file: str, world_size: int, flatten_parameters: bool):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{sync_file}", world_size=world_size, rank=gpu_id,
    )

    # Non recursive test
    model = FullyShardedDataParallel(
        nn.Sequential(nn.Linear(2, 1, bias=False), nn.Linear(1, 3, bias=False)),
        flatten_parameters=flatten_parameters
    )
    named_parameters = [(name, p.shape) for name, p in model.named_shard_view_parameters()]
    if flatten_parameters:
        if gpu_id == 0:
            assert named_parameters == [("0.weight", torch.Size([2])), ("1.weight", torch.Size([1]))]
        if gpu_id == 1:
            assert named_parameters == [("1.weight", torch.Size([2]))]
    else:
        if gpu_id == 0:
            assert named_parameters == [("0.weight", torch.Size([1])), ("1.weight", torch.Size([2]))]
        if gpu_id == 1:
            assert named_parameters == [("0.weight", torch.Size([1])), ("1.weight", torch.Size([1]))]

    # Wait before next next
    dist.barrier()

    # Recursive test
    model = nn.Sequential(
        FullyShardedDataParallel(
            nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)),
            flatten_parameters=flatten_parameters
        ),
        nn.Linear(1, 3, bias=False),
    )
    model = FullyShardedDataParallel(model, flatten_parameters=flatten_parameters)
    named_parameters = [(name, p.shape) for name, p in model.named_shard_view_parameters()]
    if flatten_parameters:
        if gpu_id == 0:
            assert named_parameters == [("1.weight", torch.Size([2])), ("0.0.weight", torch.Size([1]))]
        if gpu_id == 1:
            assert named_parameters == [("1.weight", torch.Size([1])), ("0.1.weight", torch.Size([1]))]
    else:
        if gpu_id == 0:
            assert named_parameters == [("1.weight", torch.Size([2])), ("0.0.weight", torch.Size([1])), ("0.1.weight", torch.Size([1]))]
        if gpu_id == 1:
            assert named_parameters == [("1.weight", torch.Size([1])), ("0.0.weight", torch.Size([0])), ("0.1.weight", torch.Size([0]))]


@skip_if_single_gpu
@pytest.mark.parametrize("flatten_parameters", [True, False])
def test_named_parameters_splitting(flatten_parameters: bool):
    world_size = 2
    with in_temporary_directory():
        with temp_files_ctx(num=1) as temp_files:
            mp.spawn(_check_split_worker, (temp_files[0], world_size, flatten_parameters), nprocs=world_size)


def _train_worker(gpu_id: int, sync_file: str, world_size: int, flatten_parameters: bool):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{sync_file}", world_size=world_size, rank=gpu_id,
    )

    batch_size = 4
    fake_inputs = torch.randn(size=(batch_size, 2)).cuda(gpu_id)
    fake_targets = torch.zeros(size=(batch_size, 3)).cuda(gpu_id)
    criterion = nn.MSELoss()

    model = FullyShardedDataParallel(
        nn.Sequential(nn.Linear(2, 1, bias=False), nn.Linear(1, 3, bias=False)),
        flatten_parameters=flatten_parameters
    )

    # optimizer = optim.SGD(model.shard_view_parameters(), lr=1e-2)

    num_epoch = 1
    for epoch in range(num_epoch):
        out = model(fake_inputs)
        loss = criterion(out, fake_targets)
        # optimizer.zero_grad()
        loss.backward()
        print(loss.item())
        # optimizer.step()

    for p in model.shard_view_parameters():
        print("VIEW:", p)
        print("VIEW GRAD:", p.grad)

    for p in model.parameters():
        print("PARAM:", p.data)
        print("GRAD:", p.grad)


@skip_if_single_gpu
@pytest.mark.parametrize("flatten_parameters", [True, False])
def test_named_parameters_based_training(flatten_parameters: bool):
    world_size = 2
    with in_temporary_directory():
        with temp_files_ctx(num=1) as temp_files:
            mp.spawn(_train_worker, (temp_files[0], world_size, flatten_parameters), nprocs=world_size)
