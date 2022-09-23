# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from fairscale.experimental.tooling.layer_memory_tracker import (
    LayerwiseMemoryTracker,
    ProcessGroupTracker,
    find_best_reset_points,
)
from fairscale.fair_dev.testing.testing import GPT2, dist_init, skip_if_no_cuda, skip_if_single_gpu, temp_files_ctx
from fairscale.nn import FullyShardedDataParallel


@skip_if_no_cuda()
def test_memory_tracking_traces():
    """
    Minimal test case to check that we can collect memory traces
    outside of the context of distributed training (DDP or FSDP)
    """

    # Create a model with a hierarchy of modules
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ),
        nn.Flatten(start_dim=1),
        nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True)),
    ).cuda()

    # Track a fake forward / backward
    tracker = LayerwiseMemoryTracker()
    tracker.monitor(model)
    x = torch.randn(size=(2, 3, 224, 224)).cuda()
    target = torch.LongTensor([0, 1]).cuda()
    criterion = nn.CrossEntropyLoss()
    criterion(model(x), target).backward()

    # Verify that only leaf modules are tracked and that the order
    # of the traces is consistent with backward/forward
    tracked_names = [t.module_name for t in tracker.memory_traces]
    expected_names = ["0.0", "0.1", "0.2", "0.3", "1", "2.0", "2.1"]
    assert set(expected_names) == set(tracked_names)
    assert tracked_names == (expected_names + expected_names[::-1])

    # Verify that memory tracking for ReLU is sound
    assert (
        2 * 64 * 224 * 224 * 4 == tracker.forward_traces[2].event.memory_activations
    ), "ReLU(inplace=False) should allocate activations"
    assert 0 == tracker.forward_traces[6].event.memory_activations, "ReLU(inplace=True) should NOT allocate activations"

    # Verify that overall memory tracking is sound
    summary = tracker.summary
    assert summary.total_forward_allocations >= summary.total_activation_allocations

    # Verify that the identification of top memory activation producer works:
    # these are the first layers, all allocating (2, 64, 224, 224) feature maps
    top_act_producers = summary.top_forward_activation_producers[:3]
    assert "0.0" == top_act_producers[0].module_name
    assert "0.1" == top_act_producers[1].module_name
    assert "0.2" == top_act_producers[2].module_name
    assert 3 * 3 * 64 * 3 * 4 == top_act_producers[0].module_params
    assert 64 * 2 * 4 == top_act_producers[1].module_params
    assert 0 == top_act_producers[2].module_params
    for trace in top_act_producers:
        assert 2 * 64 * 224 * 224 * 4 == trace.event.memory_activations


@skip_if_no_cuda
def test_memory_tracking_nlp_model():
    """
    Check that we can collect memory traces of a realistic model
    outside of the context of distributed training (DDP or FSDP)
    """

    BACH_SIZE = 10
    INPUT_DIM = 16
    model = GPT2(
        embed_dim=256, num_heads=2, num_layers=6, num_positions=INPUT_DIM * INPUT_DIM, num_vocab=512, num_classes=2
    ).cuda()
    tracker = LayerwiseMemoryTracker()
    tracker.monitor(model)
    input_tensor = torch.randint(10, (BACH_SIZE, INPUT_DIM)).cuda()
    output = model(input_tensor)
    output.sum().backward()

    assert len(tracker.memory_traces) > 0, "failed to collected memory traces"
    assert len(tracker.forward_traces) > 0, "failed to collect forward memory traces"
    assert len(tracker.backward_traces) > 0, "failed to collect backward memory traces"
    assert tracker.summary.total_activation_allocations == 12462080


@skip_if_single_gpu
def test_memory_tracking_ddp():
    """
    Check that we can collect memory traces of a simplistic model
    in the context of DDP distributed training
    """

    with temp_files_ctx(num=2) as sync_files:
        world_size = 2
        mp.spawn(
            _layer_memory_tracking_ddp_worker,
            (sync_files, world_size),
            nprocs=world_size,
        )


def _layer_memory_tracking_ddp_worker(gpu_id: int, sync_files: Tuple[str, str], world_size: int):
    dist_init(world_size=world_size, rank=gpu_id, filename=sync_files[0], filename_rpc=sync_files[1])
    torch.backends.cudnn.deterministic = True

    # Create different inputs on each GPU
    batch_size = 16
    torch.manual_seed(gpu_id)
    fake_inputs = torch.randn(size=(batch_size, 10)).cuda(gpu_id)
    fake_targets = torch.randn(size=(batch_size, 10)).cuda(gpu_id)
    fake_criterion = nn.MSELoss()

    # Create a simple model
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
    model = model.cuda(gpu_id)
    ddp_model = DistributedDataParallel(model, device_ids=[gpu_id])

    # Track the model on a forward / backward pass
    tracker = LayerwiseMemoryTracker()
    tracker.monitor(ddp_model)
    fake_criterion(ddp_model(fake_inputs), fake_targets).backward()
    tracker.stop()

    # Check the overall structure of the collected traces
    forward_names = [f"module.{i}" for i in range(5)]
    backward_names = [f"module.{i}" for i in reversed(range(5))]
    trace_names = [t.module_name for t in tracker.memory_traces]
    assert trace_names == (forward_names + backward_names)


@skip_if_single_gpu
def test_memory_tracking_fsdp():
    """
    Check that we can collect memory traces of a simplistic model
    in the context of FSDP distributed training
    """

    with temp_files_ctx(num=2) as sync_files:
        world_size = 2
        mp.spawn(
            _layer_memory_tracking_fsdp_worker,
            (sync_files, world_size),
            nprocs=world_size,
        )


def _layer_memory_tracking_fsdp_worker(gpu_id: int, sync_files: Tuple[str, str], world_size: int):
    dist_init(world_size=world_size, rank=gpu_id, filename=sync_files[0], filename_rpc=sync_files[1])
    torch.backends.cudnn.deterministic = True

    # Create different inputs on each GPU
    batch_size = 16
    torch.manual_seed(gpu_id)
    fake_inputs = torch.randn(size=(batch_size, 10)).cuda(gpu_id)
    fake_targets = torch.randn(size=(batch_size, 10)).cuda(gpu_id)
    fake_criterion = nn.MSELoss()

    # Create a global group and a tracker around it
    group = dist.new_group()
    group = ProcessGroupTracker(group)

    # Create a simple model
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(10, 10).cuda(gpu_id),
        nn.ReLU(),
        FullyShardedDataParallel(
            nn.Linear(10, 10).cuda(gpu_id),
            flatten_parameters=False,
            process_group=group,
        ),
        nn.ReLU(),
        FullyShardedDataParallel(
            nn.Linear(10, 10).cuda(gpu_id),
            flatten_parameters=True,
            process_group=group,
        ),
    )
    model = model.cuda(gpu_id)
    dist_model = FullyShardedDataParallel(model, flatten_parameters=False, process_group=group)

    # Track the model on a forward / backward pass
    tracker = LayerwiseMemoryTracker()
    tracker.monitor(dist_model)
    fake_criterion(dist_model(fake_inputs), fake_targets).backward()
    tracker.stop()

    # Check results of all gathers tracking (feature specific to FSDP)
    all_gathered_traces = [
        (t.module_name, t.all_gathered, t.cumul_all_gathered) for t in tracker.memory_traces if t.all_gathered > 0
    ]
    assert all_gathered_traces == [
        ("_fsdp_wrapped_module._fpw_module.0", 440, 440),
        ("_fsdp_wrapped_module._fpw_module.2._fsdp_wrapped_module._fpw_module", 440, 880),
        ("_fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module._fpw_module", 440, 880),
        ("_fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module._fpw_module", 440, 0),
        ("_fsdp_wrapped_module._fpw_module.2._fsdp_wrapped_module._fpw_module", 440, 0),
    ], all_gathered_traces


def test_find_best_reset_points():
    """
    Verify that the reset points are correctly computed
    """
    activations = [10, 8, 8, 9, 7, 7, 5, 4, 4]

    # Check boundary condition: no checkpoints
    memory, split_points = find_best_reset_points(activations, num_checkpoints=0)
    assert memory == sum(activations)

    # Check boundary condition: checkpoints everywhere
    memory, split_points = find_best_reset_points(activations, num_checkpoints=len(activations))
    assert memory == max(activations)

    # Check one checkpoint allocation
    memory, split_points = find_best_reset_points(activations, num_checkpoints=1)
    assert memory == 35
    assert split_points == [4]
    assert sum(activations[: split_points[0]]) == 35
    assert sum(activations[split_points[0] :]) == 27

    # Check multiple checkpoint allocation
    memory, split_points = find_best_reset_points(activations, num_checkpoints=2)
    assert memory == 24
    delimiters = [0] + split_points + [len(activations)]
    splits_memory = [sum(activations[i:j]) for i, j in zip(delimiters[:-1], delimiters[1:])]
    assert max(splits_memory) == memory
