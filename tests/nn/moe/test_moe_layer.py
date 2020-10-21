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

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore

if torch.cuda.is_available():
    devices = ["cpu", "cuda"]
else:
    devices = ["cpu"]

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
if "OMPI_COMM_WORLD_SIZE" in os.environ:
    dist.init_process_group(backend=dist.Backend.MPI)


def setup_module(module):
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        dist.init_process_group(backend=BACKEND, rank=0, world_size=1)


def teardown_module(module):
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("device", devices)
def test_create(device):
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert).to(device)


@pytest.mark.parametrize("device", devices)
def test_expert_params(device):
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert).to(device)
    for p in expert.parameters():
        assert p.expert is True


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_forward(device):
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


# Test Gate which round-robin routes tokens to experts
class RoundRobinGate(torch.nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts

    def forward(self, input):
        g, s, _ = input.shape
        assert s % self.num_experts == 0
        capacity = 2 * s // self.num_experts
        output = torch.zeros(g, s, self.num_experts, capacity, dtype=input.dtype, device=input.device)
        expert = 0
        c = 0
        for i in range(s):
            output[:, i, expert, c] = 1.0
            expert += 1
            if expert == self.num_experts:
                c += 1
                expert = 0

        return 0.0, output, output.bool()


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_forward_routing(device):
    model_dim = 8
    num_experts = dist.get_world_size()
    input = torch.randn(3, 4, 16, model_dim).to(device)
    gate = RoundRobinGate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use scaling matrix (each rank has a different scale)
    scale = dist.get_rank() + 1
    expert.weight = torch.nn.Parameter(torch.eye(model_dim) * scale)
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape
    # Verify that each token was sent to the correct expert by checking its scale.
    g, s, t, m = input.shape
    expert = 0
    for i in range(s):
        for j in range(t):
            for k in range(g):
                torch.allclose(input[k][i][j], output[k][i][j] * (expert + 1))
            expert += 1
            if expert == num_experts:
                expert = 0


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_backward(device):
    loss = torch.nn.MSELoss()
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
    output = loss(output, input)
    output.backward()
    assert torch.allclose(expert.weight.grad, torch.zeros_like(expert.weight))
