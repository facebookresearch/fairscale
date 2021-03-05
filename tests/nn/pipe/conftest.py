# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from typing import Any, Callable

import pytest
import torch

from fairscale.nn.model_parallel import destroy_model_parallel


@pytest.fixture(autouse=True)
def manual_seed_zero() -> None:
    torch.manual_seed(0)


def cuda_sleep_impl(seconds, cycles_per_ms):
    torch.cuda._sleep(int(seconds * cycles_per_ms * 1000))


@pytest.fixture(scope="session")
def cuda_sleep() -> Callable:
    # Warm-up CUDA.
    torch.empty(1, device="cuda")

    # From test/test_cuda.py in PyTorch.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1000000)
    end.record()
    end.synchronize()
    cycles_per_ms = 1000000 / start.elapsed_time(end)

    return functools.partial(cuda_sleep_impl, cycles_per_ms=cycles_per_ms)


def pytest_report_header() -> str:
    return f"torch: {torch.__version__}"


def pytest_runtest_setup(item: Any) -> None:
    print("setup mpi function called")


def pytest_runtest_teardown(item: Any) -> None:
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        try:
            torch.distributed.rpc.shutdown()
        except Exception:
            pass
