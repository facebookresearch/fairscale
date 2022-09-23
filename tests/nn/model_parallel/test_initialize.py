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


import torch

from fairscale.fair_dev.testing.testing import dist_init, spawn_for_all_world_sizes
from fairscale.nn.model_parallel import initialize as mpu


def run_test_initialize_model_parallel(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    if torch.distributed.get_rank() == 0:
        print("> testing initialize_model_parallel with size {} ...".format(model_parallel_size))
    model_parallel_size_ = min(model_parallel_size, torch.distributed.get_world_size())
    assert not mpu.model_parallel_is_initialized()
    mpu.initialize_model_parallel(model_parallel_size_)
    assert mpu.model_parallel_is_initialized()

    # Checks.
    def check(group, world_size, rank):
        assert world_size == torch.distributed.get_world_size(group=group)
        assert rank == torch.distributed.get_rank(group=group)

    # Model parallel.
    world_size = model_parallel_size_
    rank = torch.distributed.get_rank() % model_parallel_size_
    assert world_size == mpu.get_model_parallel_world_size()
    assert rank == mpu.get_model_parallel_rank()
    check(mpu.get_model_parallel_group(), world_size, rank)

    # Data parallel.
    world_size = torch.distributed.get_world_size() // model_parallel_size_
    rank = torch.distributed.get_rank() // model_parallel_size
    assert world_size == mpu.get_data_parallel_world_size()
    assert rank == mpu.get_data_parallel_rank()
    check(mpu.get_data_parallel_group(), world_size, rank)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(">> passed the test :-)")


def run_test_get_model_parallel_src_rank(rank, model_parallel_size_, filename, filename_rpc):
    dist_init(rank, model_parallel_size_, filename, filename_rpc)

    if torch.distributed.get_rank() == 0:
        print("> testing get_model_parallel_src_rank with size {} ...".format(model_parallel_size_))
    model_parallel_size = min(model_parallel_size_, torch.distributed.get_world_size())
    assert not mpu.model_parallel_is_initialized()
    mpu.initialize_model_parallel(model_parallel_size)
    assert mpu.model_parallel_is_initialized()

    # Checks
    src_rank = torch.distributed.get_rank() - mpu.get_model_parallel_rank()
    assert mpu.get_model_parallel_src_rank() == src_rank

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(">> passed the test :-)")


def test_initialize_model_parallel():
    spawn_for_all_world_sizes(run_test_initialize_model_parallel)


def test_get_model_parallel_src_rank():
    spawn_for_all_world_sizes(run_test_get_model_parallel_src_rank)


def test_adjacency(monkeypatch):

    new_groups = []

    data_parallel_size = 32
    pipeline_length = 8
    model_parallel_size = 4

    class MockDistribued:
        def get_rank(self):
            return 0

        def is_initialized(self):
            return True

        def get_world_size(self):
            return data_parallel_size * pipeline_length * model_parallel_size

        def new_group(self, args, backend=None):
            new_groups.append(args.copy())
            return ()

    monkeypatch.setattr(torch, "distributed", MockDistribued())

    mpu.initialize_model_parallel(model_parallel_size, pipeline_length)

    from collections import defaultdict

    buckets = defaultdict(list)

    for group in new_groups:
        buckets[len(group)].append(group)

    assert sorted(list(buckets.keys())) == [model_parallel_size, pipeline_length, data_parallel_size]

    assert len(buckets[model_parallel_size]) == pipeline_length * data_parallel_size
    assert len(buckets[data_parallel_size]) == model_parallel_size * pipeline_length
    assert len(buckets[pipeline_length]) == model_parallel_size * data_parallel_size

    # Check that model_parallel groups are contiguous
    for group in buckets[model_parallel_size]:
        assert sorted(group) == group
        assert list(range(group[0], group[-1] + 1)) == group
