# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairscale.nn import Top2Gate
from fairscale.nn.moe.top2gate import top2gating


def test_create():
    gate = Top2Gate(4, 8)


def test_forward():
    torch.manual_seed(3)
    input = torch.randn(3, 12, 4)
    gate = Top2Gate(4, 6)
    capacity = 2 * 12 // 6
    l_aux, combine_weights, dispatch_mask = gate(input)
    assert pytest.approx(l_aux.item(), 0.0283)
    assert combine_weights.shape == (3, 12, 6, 4)
    assert dispatch_mask.shape == (3, 12, 6, 4)
    assert torch.equal(combine_weights.bool(), dispatch_mask)
    assert torch.all(torch.sum(dispatch_mask, axis=(1, 3)) <= capacity)
    assert pytest.approx(torch.sum(combine_weights).item(), 36.0)


def test_top1s():
    num_tokens = 8
    num_experts = 4
    logits = torch.randn(1, num_tokens, num_experts)
    l_aux, _, dispatch_mask = top2gating(logits)
    top1s = torch.argmax(logits, dim=2)
    capacity = 2 * num_tokens // num_experts
    ce = [0] * num_experts
    locations = [0] * num_tokens
    for i, s in enumerate(top1s[0]):
        e = s.item()
        loc = ce[e]
        ce[e] = loc + 1
        if ce[e] < capacity:
            assert dispatch_mask[0][i][e][loc]
