# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairscale.nn import MOELayer, Top2Gate

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def test_create():
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert)


@skip_if_no_cuda
def test_create_cuda():
    model_dim = 8
    num_experts = 4
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim)
    moe = MOELayer(gate, expert).cuda()


def do_test_forward(device):
    model_dim = 8
    num_experts = 1
    input = torch.randn(3, 4, 16, model_dim).to(device)
    gate = Top2Gate(model_dim, num_experts)
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    # Use identity matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim))
    moe = MOELayer(gate, expert).to(device)
    output = moe(input)
    assert moe.l_aux.item() == 1.0
    assert output.shape == input.shape
    # Re-assembled output should match input due to identity expert.
    assert torch.equal(input, output)


def test_forward_cpu():
    do_test_forward("cpu")


@skip_if_no_cuda
def test_forward_cuda():
    do_test_forward("cuda")
