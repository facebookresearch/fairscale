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

import os

import pytest
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.fair_dev.testing.testing import dist_init, set_random_seed, spawn_for_all_world_sizes
from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel import layers


def run_test_parallel_embedding(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    if torch.distributed.get_rank() == 0:
        print("> testing parallel embedding with model parallel size {} ...".format(model_parallel_size))

    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    set_random_seed(123)
    input_data = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    embedding_parallel = layers.ParallelEmbedding(vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_parallel(input_data)
    loss_parallel = torch.mul(output, loss_weight).sum()
    loss_parallel.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = layers.VocabParallelEmbedding(vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    error = loss_parallel.sub(loss_original).abs()
    print("   error in loss (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    torch.distributed.barrier()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print("   error in loss (vocab parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad, hidden_size // model_parallel_size, 1)[
        mpu.get_model_parallel_rank()
    ]
    error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad, vocab_size // model_parallel_size, 0)[
        mpu.get_model_parallel_rank()
    ]
    error = embedding_vocab_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print("   error in grad (vocab parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(">> passed the test :-)")


def run_test_initialize_affine_weight(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing initialize_affine_weight with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    layers._initialize_affine_weight(weight, output_size, input_size, output_size_coeff, 0, torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff, dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print(
        "   column parallel max error (should be zero) on global rank {}: {}".format(
            torch.distributed.get_rank(), error
        )
    )
    assert error < 1.0e-6, error

    # ------------
    # Row parallel
    # ------------
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    layers._initialize_affine_weight(weight, output_size, input_size, input_size_coeff, 1, torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff, dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print(
        "   row parallel max error (should be zero) on global rank {}: {}".format(torch.distributed.get_rank(), error)
    )
    assert error < 1.0e-6, error

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def run_test_column_parallel_linear(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing ColumnParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = layers.ColumnParallelLinear(input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdA on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    my_dLdb = torch.split(dLdb, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdb on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdX on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


def run_test_row_parallel_linear(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing RowParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = layers.RowParallelLinear(input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff, dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdA on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdb on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdX on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


def test_affine_weight():
    spawn_for_all_world_sizes(run_test_initialize_affine_weight, deterministic=True)


def test_embedding():
    spawn_for_all_world_sizes(run_test_parallel_embedding, deterministic=True)


def test_column_parallel():
    spawn_for_all_world_sizes(run_test_column_parallel_linear, deterministic=True)


@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="only works on mpi")
def test_row_parallel():
    spawn_for_all_world_sizes(run_test_row_parallel_linear, deterministic=True)
