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
