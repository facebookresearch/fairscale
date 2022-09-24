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
import torch.nn.functional as F

from fairscale.fair_dev.testing.testing import IdentityLayer, dist_init, set_random_seed, spawn_for_all_world_sizes
from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.cross_entropy import vocab_parallel_cross_entropy
from fairscale.nn.model_parallel.mappings import scatter_to_model_parallel_region


def torch_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed):
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size), scale=logits_scale).cuda()
    logits = identity()
    target = torch.cuda.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size()[-1]), target.view(-1), reduction="none").view_as(target).mean()
    loss.backward()
    return loss, identity.weight.grad


def mpu_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed):
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size), scale=logits_scale).cuda()
    logits = identity()
    logits_parallel = scatter_to_model_parallel_region(logits)
    target = torch.cuda.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = vocab_parallel_cross_entropy(logits_parallel, target).mean()
    loss.backward()
    return loss, identity.weight.grad


def run_test_cross_entropy(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    if torch.distributed.get_rank() == 0:
        print("> testing cross entropy with model parallel size {} ...".format(model_parallel_size))

    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    batch_size = 13
    seq_length = 17
    vocab_size_per_partition = 11
    logits_scale = 1000.0
    vocab_size = vocab_size_per_partition * model_parallel_size
    seed = 1234

    loss_torch, grad_torch = torch_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed)
    loss_mpu, grad_mpu = mpu_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed)

    error = loss_torch.sub_(loss_mpu).abs().max()
    print("   max error in loss on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = grad_torch.sub_(grad_mpu).abs().max()
    print("   max error in grad on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(">> passed the test :-)")


def test_cross_entropy():
    spawn_for_all_world_sizes(run_test_cross_entropy)
