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
from unittest.mock import patch

import pytest
import torch
from torch.cuda import Event
import torch.multiprocessing as mp
import torch.nn as nn

from fair_dev.testing.testing import dist_init, get_cycles_per_ms, skip_if_single_gpu, teardown, temp_files_ctx
from fairscale.internal import torch_version
from fairscale.nn import enable_wrap, wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


class Layer(nn.Module):
    def __init__(self, compute_cycles, has_params: bool):
        super().__init__()
        self.sleep_cycles = compute_cycles
        self.optional_param = None
        if has_params:
            self.optional_param = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # Get 2 events.
        self.e1 = Event(enable_timing=True)
        self.e2 = Event(enable_timing=True)

        # Record the fake forward compute time.
        self.e1.record()
        if self.sleep_cycles > 0:
            torch.cuda._sleep(self.sleep_cycles)
        if self.optional_param is not None:
            x = x + self.optional_param  # force the param to be part of the graph
        self.e2.record()
        return x

    def get_time(self):
        # return the recorded duration.
        return self.e1.elapsed_time(self.e2)


def _create_model(fsdp_config, compute_cycles, has_params: bool):
    with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
        model = wrap(
            nn.Sequential(
                wrap(Layer(compute_cycles, has_params)),
                wrap(Layer(compute_cycles, has_params)),
                wrap(Layer(compute_cycles, has_params)),
                wrap(Layer(compute_cycles, has_params)),
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
    gpu_id,
    world_size,
    fsdp_config,
    tempfile,
    tempfile_rpc,
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, tempfile, tempfile_rpc)
    assert result, "Dist init failed"

    # Save the original torch.distributed.all_gather function since we will
    # patch it to include an artificial delay.
    orig_all_gather = torch.distributed.all_gather
    orig_all_gather_base = (
        torch.distributed._all_gather_base if hasattr(torch.distributed, "_all_gather_base") else None
    )

    def run(compute_cycles, all_gather_cycles):
        has_params = all_gather_cycles > 0
        model = _create_model(fsdp_config, compute_cycles, has_params)

        # Get the input and sets the input's requires_grad to True because
        # we have a fake compute in the forward pass.
        batch = torch.rand(1).cuda()
        batch.requires_grad = True

        # We run 20 iterations but only collect timing data from the minimal 10
        # data points because nondeterministic system events can disturb the timing.
        gpu_compute = Min10()
        gpu_total = Min10()
        for _ in range(20):
            # Get two events for measuring the overall time.
            e1 = Event(enable_timing=True)
            e2 = Event(enable_timing=True)

            cpu_start = time.process_time()

            all_gather_called = False
            all_gather_base_called = False

            def _delayed_all_gather(*args, **kwargs):
                nonlocal all_gather_called
                all_gather_called = True
                torch.cuda._sleep(all_gather_cycles)
                return orig_all_gather(*args, **kwargs)

            def _delayed_all_gather_base(*args, **kwargs):
                nonlocal all_gather_base_called
                all_gather_base_called = True
                torch.cuda._sleep(all_gather_cycles)
                assert orig_all_gather_base
                return orig_all_gather_base(*args, **kwargs)

            method_string_all_gather_base = "torch.distributed._all_gather_base"
            if hasattr(torch.distributed, "_all_gather_base") is False:
                # no such method, to make mock_all_gather_base 0 invocation, use an impossible name
                method_string_all_gather_base = "math.nan"
                pass
            # forward pass
            #
            # Even though both e1 & e2 are on the compute stream, since
            # compute depends on all_gather, e2-e1 includes all_gather time.
            e1.record()
            with patch("torch.distributed.all_gather", _delayed_all_gather):
                with patch(method_string_all_gather_base, _delayed_all_gather_base):
                    out = model(batch)
                    if has_params and world_size > 1:
                        assert all_gather_called or all_gather_base_called
                    else:
                        assert not all_gather_called and not all_gather_base_called
            e2.record()

            # backward pass
            out.backward()
            if torch_version() >= (1, 7, 0):
                model.zero_grad(set_to_none=True)
            else:
                for p in model.parameters():
                    p.grad = None
            out.item()

            # get sum of the compute time
            times = []
            for mod in model.modules():
                if not isinstance(mod, Layer):
                    continue
                times.append(mod.get_time())

            # get gpu compute + all_gather time
            overall_gpu_time = e1.elapsed_time(e2)

            gpu_compute.add(sum(times))
            gpu_total.add(overall_gpu_time)

        del model
        return {
            "gpu_compute": gpu_compute.avg(),
            "gpu_total": gpu_total.avg(),
        }

    sleep_cycles = int(100 * get_cycles_per_ms())

    e1 = run(0, 0)  # no compute, no all-gather
    e2 = run(0, sleep_cycles)  # no compute, only all-gather
    e3 = run(sleep_cycles, 0)  # only compute, no all-gather
    e4 = run(sleep_cycles, sleep_cycles)  # both compute and all-gather
    debug_string = f"\nrank{rank}:\n  e1: {e1}\n  e2: {e2}\n  e3: {e3}\n  e4: {e4}"
    print(debug_string)

    # Check the GPU timing.
    short = [e1["gpu_compute"], e1["gpu_total"], e2["gpu_compute"]]
    long = [e3["gpu_compute"], e3["gpu_total"], e4["gpu_compute"], e4["gpu_total"]]
    if world_size == 1:
        short.append(e2["gpu_total"])  # all gather should not be happening.
    else:
        long.append(e2["gpu_total"])  # all gather should happen and prolong the gpu wait.
    for s in short:
        for l in long:
            # 10X longer is a safe margin, since the time is around 100X longer
            # when there is compute work on GPU vs. no work.
            assert s * 10 < l, f"{s} * 10 < {l} in " + debug_string

    # Check the GPU overlapping when there is all-gather.
    if world_size > 1:
        compute_only = e3["gpu_compute"]
        all_gather_only = e2["gpu_total"]
        both = e4["gpu_total"]
        assert compute_only + all_gather_only > 1.1 * both, (
            f"{compute_only} + {all_gather_only} > 1.1 * {both} in " + debug_string
        )

    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
@pytest.mark.parametrize("mixed", ["mixed", "full"])
def test_forward_overlap(world_size, flatten, mixed):
    fsdp_config = {
        "flatten_parameters": flatten == "flatten",
        "mixed_precision": mixed == "mixed",
    }
    with temp_files_ctx(2) as temp_files:
        mp.spawn(
            _distributed_worker,
            (world_size, fsdp_config, temp_files[0], temp_files[1]),
            nprocs=world_size,
        )
