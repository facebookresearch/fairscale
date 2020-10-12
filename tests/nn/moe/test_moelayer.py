# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist

from fairscale.nn import MOELayer, Top2Gate

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
dist.init_process_group(backend=dist.Backend.MPI)


def test_create():
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert)


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_forward_multi(device):
    model_dim = 8
    num_experts = dist.get_world_size(dist.group.WORLD)
    input = torch.randn(3, 4, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use identity matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim))
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape
    # Re-assembled output should match input due to identity expert.
    assert torch.allclose(input, output)
