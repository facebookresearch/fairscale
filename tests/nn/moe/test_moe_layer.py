# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fairscale.fair_dev.testing.testing import make_cudnn_deterministic
from fairscale.internal import torch_version
from fairscale.nn import MOELayer, Top2Gate

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_version() >= (1, 8, 0)), reason="cuda and torch>=1.8.0 required"
)

devices = ["cuda"]


def pg_worker(rank, world_size, init_file, func, *args):
    init_url = "file://" + init_file
    dist.init_process_group(backend=dist.Backend.NCCL, rank=rank, world_size=world_size, init_method=init_url)
    torch.cuda.set_device(rank)
    dist.all_reduce(torch.zeros(1).cuda())
    func(*args)
    dist.destroy_process_group()


def pg_test(world_size=torch.cuda.device_count()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tempfile_name = tempfile.mkstemp()[1]
            mp.spawn(pg_worker, args=(world_size, tempfile_name, func, *kwargs.values()), nprocs=world_size)

        globals()["test_" + func.__name__] = wrapper
        return func

    return decorator


@pg_test(world_size=1)
@pytest.mark.parametrize("device", devices)
def create(device):
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert).to(device)


@pg_test(world_size=1)
@pytest.mark.parametrize("device", devices)
def expert_params(device):
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert).to(device)
    for p in expert.parameters():
        assert p.expert is True, str(p.expert)


@pg_test()
@pytest.mark.parametrize("device", devices)
def forward(device):
    make_cudnn_deterministic()
    model_dim = 8
    num_experts = dist.get_world_size(dist.group.WORLD)
    input = torch.randn(4, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use identity matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim))
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape, f"{output.shape} != {input.shape}"
    # Re-assembled output should match input due to identity expert.
    torch.testing.assert_allclose(input, output)


@pg_test()
@pytest.mark.parametrize("device", devices)
def forward_multi(device):
    make_cudnn_deterministic()
    torch.set_printoptions(threshold=5000)
    num_local_experts = 4
    model_dim = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.randn(4 * num_local_experts, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    experts = []
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # Use identity matrix
        expert.weight = torch.nn.Parameter(torch.eye(model_dim))
        experts += [expert]
    moe = MOELayer(gate, torch.nn.ModuleList(experts)).to(device)
    output = moe(input)
    assert output.shape == input.shape, f"{output.shape} != {input.shape}"
    # 90% of the input should have gone to an expert
    assert (
        len(output.nonzero(as_tuple=False)) / output.numel() > 0.90
    ), f"{len(output.nonzero(as_tuple=False))} / {output.numel()}"
    # Except for zeros, re-assembled output should match input due to identity expert.
    torch.testing.assert_allclose(input, torch.where(output > 0, output, input))


# Test Gate which round-robin routes tokens to experts
class RoundRobinGate(torch.nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts

    def forward(self, input):
        s = input.shape[0]
        assert s % self.num_experts == 0, f"{s} % {self.num_experts} != 0"
        capacity = 2 * s // self.num_experts
        output = torch.zeros(s, self.num_experts, capacity, dtype=input.dtype, device=input.device)
        for i in range(s):
            output[i, i % self.num_experts, i // self.num_experts] = 1.0
        return 0.0, output, output.bool()


@pg_test()
@pytest.mark.parametrize("device", devices)
def forward_routing(device):
    make_cudnn_deterministic()
    model_dim = 8
    num_experts = dist.get_world_size()
    input = torch.randn(4, 16, model_dim).to(device)
    gate = RoundRobinGate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use scaling matrix (each rank has a different scale)
    scale = dist.get_rank() + 1
    expert.weight = torch.nn.Parameter(torch.eye(model_dim) * scale)
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape, f"{output.shape} != {input.shape}"
    # Verify that each token was sent to the correct expert by checking its scale.
    t = input.shape[1]
    for i in range(t):
        expert = i % num_experts
        torch.testing.assert_allclose(input[:, i] * (expert + 1), output[:, i])


@pg_test()
@pytest.mark.parametrize("device", devices)
def forward_routing_multi(device):
    make_cudnn_deterministic()
    model_dim = 8
    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.randn(4 * num_local_experts, 16, model_dim).to(device)
    gate = RoundRobinGate(model_dim, num_experts)
    experts = []
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # Use scaling matrix (each rank has a different scale)
        scale = dist.get_rank() * num_local_experts + i + 1
        expert.weight = torch.nn.Parameter(torch.eye(model_dim) * scale)
        experts += [expert]
    moe = MOELayer(gate, torch.nn.ModuleList(experts)).to(device)
    output = moe(input)
    assert output.shape == input.shape, f"{output.shape} != {input.shape}"
    # Verify that each token was sent to the correct expert by checking its scale.
    t = input.shape[1]
    for i in range(t):
        expert = i % num_experts
        torch.testing.assert_allclose(input[:, i] * (expert + 1), output[:, i])


@pg_test()
@pytest.mark.parametrize("device", devices)
def backward(device):
    make_cudnn_deterministic()
    loss = torch.nn.MSELoss()
    model_dim = 8
    num_experts = dist.get_world_size(dist.group.WORLD)
    input = torch.randn(4, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use identity matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim))
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert output.shape == input.shape, f"{output.shape} != {input.shape}"
    output = loss(output, input)
    output.backward()
    torch.testing.assert_allclose(expert.weight.grad, torch.zeros_like(expert.weight))
