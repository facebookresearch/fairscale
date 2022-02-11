# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairscale.nn import Top2Gate
from fairscale.nn.moe.top2gate import top2gating

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def test_create():
    gate = Top2Gate(4, 8)


@skip_if_no_cuda
def test_create_cuda():
    gate = Top2Gate(4, 8).cuda()


def do_test_forward(device):
    torch.manual_seed(3)
    input = torch.randn(12, 4).to(device)
    gate = Top2Gate(4, 6).to(device)
    capacity = 2 * 12 // 6
    l_aux, combine_weights, dispatch_mask = gate(input)
    assert pytest.approx(l_aux.item(), rel=0.01) == 0.0267, l_aux
    assert combine_weights.shape == (12, 6, 4)
    assert dispatch_mask.shape == (12, 6, 4)
    assert torch.equal(combine_weights.bool(), dispatch_mask)
    assert torch.all(torch.sum(dispatch_mask, axis=(0, 2)) <= capacity)
    assert torch.all(combine_weights >= 0.0)
    assert torch.all(combine_weights <= 1.0)
    weights_sum = torch.sum(combine_weights).item()
    assert round(weights_sum) == pytest.approx(weights_sum), weights_sum
    # For this random seed, we get 12 slots filled.
    assert weights_sum == pytest.approx(12.0), weights_sum


def test_forward_cpu():
    do_test_forward("cpu")


@skip_if_no_cuda
def test_forward_cuda():
    do_test_forward("cuda")


# Verify that top gate is allocated capacity as per Algorithm 1 in GShard paper.
def test_expert1_overflow():
    num_tokens = 8
    num_experts = 4
    logits = torch.randn(num_tokens, num_experts)
    logits[:, 0] = torch.max(logits, dim=1).values + 1  # Force overflow
    top1s = torch.argmax(logits, dim=1)
    assert top1s.eq(0).all(), top1s
    _, __, dispatch_mask = top2gating(logits)
    capacity = 2 * num_tokens // num_experts

    for i in range(num_tokens):
        if i < capacity:
            assert dispatch_mask[i][0][i]
        else:
            assert not dispatch_mask[i][0].any()
