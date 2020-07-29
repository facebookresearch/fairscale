# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import functools

import pytest
import torch

try:
    from fairscale.optim.adam import Adam

    imported_adam = True
except ImportError:
    imported_adam = False

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_no_adam = pytest.mark.skipif(not imported_adam, reason="Fairscale Adam not available")


@skip_if_no_cuda
@skip_if_no_adam
def test_step():
    weight = torch.randn(10, 5).cuda().requires_grad_()
    bias = torch.randn(10).cuda().requires_grad_()
    input = torch.randn(5).cuda()
    optimizer = Adam([weight, bias], lr=1e-3)

    # to check if the optimizer can be printed as a string
    optimizer.__repr__()

    def fn():
        optimizer.zero_grad()
        y = weight.mv(input)
        if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
            y = y.cuda(bias.get_device())
        loss = (y + bias).pow(2).sum()
        loss.backward()
        return loss

    initial_value = fn().item()
    for _i in range(5):
        optimizer.step(fn)
    assert fn().item() < initial_value


@skip_if_no_cuda
@skip_if_no_adam
def test_step_multigpu():
    if not torch.cuda.device_count() > 1:
        return
    weight = torch.randn(10, 5).cuda(0).requires_grad_()
    bias = torch.randn(10).cuda(1).requires_grad_()
    input = torch.randn(5).cuda(0)
    optimizer = Adam([weight, bias], lr=1e-3)

    # to check if the optimizer can be printed as a string
    optimizer.__repr__()

    def fn():
        optimizer.zero_grad()
        y = weight.mv(input)
        if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
            y = y.cuda(bias.get_device())
        loss = (y + bias).pow(2).sum()
        loss.backward()
        return loss

    initial_value = fn().item()
    for _i in range(5):
        optimizer.step(fn)
    assert fn().item() < initial_value


@skip_if_no_cuda
@skip_if_no_adam
def test_state_dict():
    weight = torch.randn(10, 5).float().cuda().requires_grad_()
    bias = torch.randn(10).float().cuda().requires_grad_()
    input = torch.randn(5).float().cuda()

    optimizer = Adam([weight, bias], lr=1e-3)

    def fn_base(optimizer, weight, bias, input):
        optimizer.zero_grad()
        loss = (weight.mv(input) + bias).pow(2).sum()
        loss.backward()
        return loss

    fn = functools.partial(fn_base, optimizer, weight, bias, input)

    # Prime the optimizer
    for _i in range(5):
        optimizer.step(fn)
    # Clone the weights and construct new optimizer for them
    weight_c = weight.data.clone().requires_grad_()
    bias_c = bias.data.clone().requires_grad_()
    optimizer_c = Adam([weight_c, bias_c], lr=1e-3)
    fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c, input)
    # Load state dict
    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer.state_dict())
    optimizer_c.load_state_dict(state_dict_c)
    # Run both optimizations in parallel
    for _i in range(5):
        optimizer.step(fn)
        optimizer_c.step(fn_c)
        assert torch.equal(weight, weight_c)
        assert torch.equal(bias, bias_c)


@skip_if_no_cuda
@skip_if_no_adam
def test_invalid_beta():
    weight = torch.randn(10, 5, requires_grad=True).float().cuda()
    bias = torch.randn(10, requires_grad=True).float().cuda()
    with pytest.raises(ValueError):
        Adam([weight, bias], lr=1e-2, betas=(1.0, 0.0))


@skip_if_no_cuda
@skip_if_no_adam
def test_invalid_weight_decay():
    weight = torch.randn(10, 5, requires_grad=True).float().cuda()
    bias = torch.randn(10, requires_grad=True).float().cuda()
    with pytest.raises(ValueError):
        Adam([weight, bias], lr=1e-2, weight_decay=-1)


@skip_if_no_cuda
@skip_if_no_adam
def test_amsgrad():
    weight = torch.randn(10, 5, requires_grad=True).float().cuda()
    bias = torch.randn(10, requires_grad=True).float().cuda()
    with pytest.raises(RuntimeError):
        Adam([weight, bias], lr=1e-2, amsgrad=True)
