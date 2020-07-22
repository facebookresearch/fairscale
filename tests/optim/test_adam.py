# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairscale.optim.adam import FusedAdamV1 as Adam

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


@skip_if_no_cuda
def test_step():
    device = "cuda"
    x_val = 2
    weight = 3.0
    bias = 5.0
    error = 1.0
    target = torch.tensor([x_val * weight + bias + error], device=device)
    loss_fn = torch.nn.L1Loss()

    x = torch.tensor([float(x_val)], device=device)
    m = torch.nn.Linear(1, 1)
    m.weight.data = torch.tensor([[weight]])
    m.bias.data = torch.tensor([bias])
    m.to(device)
    o = Adam(m.parameters())
    y = m(x)
    y.backward(x)

    def closure():
        o.zero_grad()
        output = m(x)
        loss = loss_fn(output, target)
        loss.backward()
        return loss

    loss = o.step(closure=closure)

    assert loss == torch.tensor(error, device=device)
    assert m.weight == torch.tensor([[3.001]], device=device)
    assert m.bias == torch.tensor([5.001], device=device)
