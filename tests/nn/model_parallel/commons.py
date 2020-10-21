# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import multiprocessing
import os
import random

import numpy
from packaging import version
import pytest
import torch
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp

from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed


class IdentityLayer(torch.nn.Module):
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)


def dist_init(rank, world_size, hostname=None):
    if hostname is None:
        hostname = "localhost"
    print(f"dist init r={rank}, world={world_size}, host={hostname}")
    os.environ["MASTER_ADDR"] = hostname
    os.environ["MASTER_PORT"] = "10638"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    if version.parse(torch.__version__).release >= (1, 6, 0):
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=init_method)
        os.environ["MASTER_ADDR"] = hostname
        os.environ["MASTER_PORT"] = "10639"
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        rpc.init_rpc(
            f"Test{rank}",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
        )
    else:
        if world_size > 1:
            rpc.init_rpc(f"Test{rank}", rank=rank, world_size=world_size)
        else:
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if torch.cuda.is_available() and torch.cuda.device_count():
        torch.cuda.set_device(rank % torch.cuda.device_count())


def get_worker_map():
    return {rank: f"Test{rank}" for rank in range(dist.get_world_size())}


def get_world_sizes():
    limit = torch.cuda.device_count()
    return [x for x in [1, 2, 4, 8] if x <= limit]


def spawn_for_all_world_sizes(test_func, world_sizes=get_world_sizes(), args=[]):
    for world_size in world_sizes:
        mp.spawn(test_func, args=(world_size, *args), nprocs=world_size, join=True)


def worker_process(rank, world_size, func, args, error_queue):
    """Main function for unit tests launced with torch_spawn"""

    dist_init(rank, world_size)
    kwargs = {}
    if "OMPI_COMM_WORLD_RANK" not in os.environ:
        kwargs["pipeline_backend"] = "gloo"
    initialize_model_parallel(1, world_size, **kwargs)
    try:
        func(*args)
    except BaseException as e:
        # If the function raises 'Skipped', this indicates pytest.skip(), so
        # forward it to parent so we can call pytest.skip() there
        if e.__class__.__name__ == "Skipped":
            error_queue.put(str(e))
            return
        raise e


def torch_spawn(world_sizes=None):
    if world_sizes is None:
        world_sizes = get_world_sizes()

    def prepare_test(func):
        """Function called with the test function as the argument. Generates a
        replacement which serves as the actual test function."""

        name = func.__name__
        parameters = inspect.signature(func).parameters

        if name.startswith("test"):
            raise ValueError(
                f"Tests marked with @torch_spawn (i.e. '{name}') should not have names beginning in 'test' as they will"
                " be picked up by pytest without running the spawn wrapper"
            )

        @functools.wraps(func)
        def replacement(*args, **kwargs):
            assert args == tuple()
            args = tuple(
                kwargs[p] for p in parameters if p != "rank"
            )  # converting named parameters to positional parameters to pass to `spawn`

            error_queue = multiprocessing.get_context("spawn").SimpleQueue()
            if "OMPI_COMM_WORLD_RANK" in os.environ:
                os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
                os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "10638"
                torch.distributed.init_process_group("mpi")
                world_size = torch.distributed.get_world_size()
                initialize_model_parallel(1, world_size)
                torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
                if world_size in world_sizes:
                    func(*args)
                else:
                    pytest.skip(f"requested world size doesn't match current world size")
            else:
                spawn_for_all_world_sizes(worker_process, world_sizes, (func, args, error_queue))

            if not error_queue.empty():
                msg = error_queue.get()
                pytest.skip(msg)

        # Register a function with the same name, prefixed with "test_" in the
        # calling module, so it will be picked up by pytest
        caller_module = inspect.getmodule(inspect.currentframe().f_back)
        setattr(caller_module, f"test_{name}", replacement)

        return func

    return prepare_test
