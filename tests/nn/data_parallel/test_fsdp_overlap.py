# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP and ensure expected overlapping between all_gather and forward. """

from statistics import mean
import time

import pytest
import torch
from torch.cuda import Event
import torch.multiprocessing as mp
import torch.nn as nn

from fairscale.nn import enable_wrap, wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, temp_files_ctx, torch_version


class Layer(nn.Module):
    def __init__(self, compute_cycles, all_gather_mb):
        super().__init__()
        self.sleep_cycles = compute_cycles
        if all_gather_mb > 0:
            self.l = nn.Linear(all_gather_mb * 1024 * 1024 // 4, 1)

    def forward(self, x):
        # Get 2 events.
        self.e1 = Event(enable_timing=True)
        self.e2 = Event(enable_timing=True)

        # Record the fake forward compute time.
        self.e1.record()
        torch.cuda._sleep(self.sleep_cycles)
        self.e2.record()
        return x

    def get_time(self):
        # return the recorded duration.
        return self.e1.elapsed_time(self.e2)


def _create_model(fsdp_config, compute_cycles, all_gather_mb):
    with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
        model = wrap(
            nn.Sequential(
                wrap(Layer(compute_cycles, all_gather_mb)),
                wrap(Layer(compute_cycles, all_gather_mb)),
                wrap(Layer(compute_cycles, all_gather_mb)),
                wrap(Layer(compute_cycles, all_gather_mb)),
            )
        ).cuda()
    return model


class Min10:
    def __init__(self):
        self.data = []

    def add(self, new_data):
        if len(self.data) < 10:
            self.data.append(new_data)
        else:
            self.data = sorted(self.data)
            if new_data < self.data[-1]:
                self.data[-1] = new_data

    def avg(self):
        return mean(self.data)


def _distributed_worker(
    gpu_id, world_size, fsdp_config, tempfile, tempfile_rpc,
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, tempfile, tempfile_rpc)
    assert result, "Dist init failed"

    def run(compute_cycles, all_gather_mb):
        model = _create_model(fsdp_config, compute_cycles, all_gather_mb)

        if gpu_id == 0:
            print(model)

        # Get the input and sets the input's requires_grad to True because
        # we have a fake compute in the forward pass.
        batch = torch.rand(1).cuda()
        batch.requires_grad = True

        # We run 20 iterations but only collect timing data from the minimal 10
        # data points because nondeterministic system events can disturbe the timing.
        cpu_iter = Min10()
        cpu_wait = Min10()
        gpu_compute = Min10()
        gpu_total = Min10()
        for _ in range(20):
            # Two events for measuring the overall time.
            e1 = Event(enable_timing=True)
            e2 = Event(enable_timing=True)

            cpu_start = time.process_time()

            # forward
            e1.record()
            out = model(batch)
            e2.record()

            # backward
            out.backward()
            if torch_version() >= (1, 7, 0):
                model.zero_grad(set_to_none=True)
            else:
                for p in model.parameters():
                    p.grad = None

            cpu_iter_time = time.process_time() - cpu_start

            # wait for gpu
            out.item()
            cpu_wait_for_gpu_time = time.process_time() - cpu_start - cpu_iter_time

            # get sum of compute time
            times = []
            for mod in model.modules():
                if not isinstance(mod, Layer):
                    continue
                times.append(mod.get_time())

            # get gpu compute + all_gather time
            overall_gpu_time = e1.elapsed_time(e2)

            # print(f"rank {rank}", cpu_iter_time, cpu_wait_for_gpu_time, sum(times), "vs", e1.elapsed_time(e2))
            cpu_iter.add(cpu_iter_time)
            cpu_wait.add(cpu_wait_for_gpu_time)
            gpu_compute.add(sum(times))
            gpu_total.add(e1.elapsed_time(e2))

        return [cpu_iter.avg(), cpu_wait.avg(), gpu_compute.avg(), gpu_total.avg()]

    compute_cycles = 100_000_000
    data_mb = 10
    if fsdp_config["mixed_precision"]:
        # make sure all-gather amount are the same in both mixed and full.
        data_mb *= 2

    e1 = run(0, 0)  # no compute, no all-gather
    e2 = run(0, data_mb)  # no compute, only all-gather
    e3 = run(compute_cycles, 0)  # only compute, no all-gather
    e4 = run(compute_cycles, data_mb)  # both compute and all-gather
    print(f"rank{rank}:\n  {e1}\n  {e2}\n  {e3}\n  {e4}")

    # Check the cpu/gpu timing. CPU should run ahead of GPU. Therefore, cpu-gpu
    # wait should be long, except when there is no real work on GPU.
    #
    # If the assertions fail below, we likely have a cpu-gpu wait in the forward/backward pass.
    short = [e1[0], e2[0], e3[0], e4[0], e1[1]]
    long = [e3[1], e4[1]]
    if world_size == 1:
        short.append(e2[1])  # all gather should not be happening.
    else:
        long.append(e2[1])  # all gather should happen and prolong the cpu-gpu wait.
    for s in short:
        for l in long:
            # 10X longer is a safe margin, since the GPU work timing is around 100X more
            # of that of the CPU.
            assert s * 10 < l, f"{s} and {l}"

    # Check the GPU timing.
    short = [e1[2], e1[3], e2[2]]
    long = [e3[2], e3[3], e4[2], e4[3]]
    if world_size == 1:
        short.append(e2[3])  # all gather should not be happening.
    else:
        long.append(e2[3])  # all gather should happen and prolong the cpu-gpu wait.
    for s in short:
        for l in long:
            # 10X longer is a safe margin, since the time is around 100X longer
            # when there is work on GPU vs. no work.
            assert s * 10 < l, f"{s} and {l}"

    # Check the GPU overlapping when there is all-gather.
    if world_size > 1:
        compute_only = e3[2]
        all_gather_only = e2[3]
        both = e4[3]
        assert compute_only + all_gather_only > 1.1 * both, f"{compute_only} {all_gather_only} > 1.1 {both}"

    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("flatten", [True, False])
@pytest.mark.parametrize("mixed", [True, False])
def test_forward_overlap(world_size, flatten, mixed):
    fsdp_config = {
        "flatten_parameters": flatten,
        "mixed_precision": mixed,
    }
    with temp_files_ctx(2) as temp_files:
        mp.spawn(
            _distributed_worker, (world_size, fsdp_config, temp_files[0], temp_files[1]), nprocs=world_size,
        )
