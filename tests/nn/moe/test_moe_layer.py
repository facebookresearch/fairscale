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
    pass  # dist.init_process_group(backend=dist.Backend.MPI)


def setup_module(module):
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        dist.init_process_group(backend=BACKEND, rank=0, world_size=1)
    else:
        dist.init_process_group(backend=dist.Backend.MPI)


def teardown_module(module):
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
    input = torch.randn(1, 4, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use identity matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim))
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape
    # Re-assembled output should match input due to identity expert.
    assert torch.allclose(input, output)


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_forward_multi(device):
    torch.set_printoptions(threshold=5000)
    num_local_experts = 4
    model_dim = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.randn(num_local_experts, 4, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    experts = []
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # Use identity matrix
        expert.weight = torch.nn.Parameter(torch.eye(model_dim))
        experts += [expert]
    moe = MOELayer(gate, torch.nn.ModuleList(experts)).to(device)
    output = moe(input)
    assert output.shape == input.shape
    # 90% of the input should have gone to an expert
    assert len(output.nonzero(as_tuple=False)) / output.numel() > 0.90
    # Except for zeros, re-assembled output should match input due to identity expert.
    assert torch.allclose(input, torch.where(output > 0, output, input))


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
        for i in range(s):
            output[:, i, i % self.num_experts, i // self.num_experts] = 1.0
        return 0.0, output, output.bool()


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_forward_routing(device):
    model_dim = 8
    num_experts = dist.get_world_size()
    input = torch.randn(1, 4, 16, model_dim).to(device)
    gate = RoundRobinGate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use scaling matrix (each rank has a different scale)
    scale = dist.get_rank() + 1
    expert.weight = torch.nn.Parameter(torch.eye(model_dim) * scale)
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape
    # Verify that each token was sent to the correct expert by checking its scale.
    t = input.shape[2]
    for i in range(t):
        expert = i % num_experts
        assert torch.allclose(input[:, :, i] * (expert + 1), output[:, :, i])


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu"])
def test_backward(device):
    loss = torch.nn.MSELoss()
    model_dim = 8
    num_experts = dist.get_world_size(dist.group.WORLD)
    input = torch.randn(1, 4, 16, model_dim).to(device)
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
