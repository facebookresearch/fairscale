# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import functools

import pytest
import torch

from fairscale.optim.grad_scaler import FairscaleGradScaler

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
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                assert p.dtype == torch.float32
    with pytest.raises(AttributeError):
        optimizer.fp32_param_groups


@skip_if_no_cuda
@skip_if_no_adam
def test_step_me():
    weight = torch.randn(10, 5).cuda().half().requires_grad_()
    bias = torch.randn(10).cuda().half().requires_grad_()
    input = torch.randn(5).half().cuda()
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
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                assert p.dtype == torch.float16
    with pytest.raises(AttributeError):
        optimizer.fp32_param_groups


@skip_if_no_cuda
@skip_if_no_adam
def test_step_mixed_precision():
    weight = torch.randn(10, 5).cuda().half().requires_grad_()
    bias = torch.randn(10).cuda().half().requires_grad_()
    input = torch.randn(5).half().cuda()
    optimizer = Adam([weight, bias], lr=1e-3, mixed_precision=True)

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
    assert len(optimizer.fp32_param_groups) == len(optimizer.param_groups)

    for fp32_group, fp16_group in zip(optimizer.fp32_param_groups, optimizer.param_groups):
        for fp32_p, fp16_p in zip(fp32_group["params"], fp16_group["params"]):

            def assert_almost_zero(x):
                assert abs(x) < 1e-3
                return 1.0

            assert fp32_p.dtype == torch.float32
            if fp16_p.requires_grad:
                assert fp16_p.dtype == torch.float16
                (fp32_p - fp16_p).to("cpu").detach().apply_(assert_almost_zero)


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
def test_step_multigpu_mixed_precision():
    if not torch.cuda.device_count() > 1:
        return
    weight = torch.randn(10, 5).cuda(0).half().requires_grad_()
    bias = torch.randn(10).cuda(1).half().requires_grad_()
    input = torch.randn(5).cuda(0).half()
    optimizer = Adam([weight, bias], lr=1e-3, mixed_precision=True)

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
def test_with_grad_scaler():
    weight = torch.randn(10, 5).cuda().half().requires_grad_()
    bias = torch.randn(10).cuda().half().requires_grad_()
    input = torch.randn(5).half().cuda()
    target = torch.randn(10).half().cuda()
    optimizer = Adam([weight, bias], lr=1e-3, mixed_precision=True)
    scaler = FairscaleGradScaler()
    criterion = torch.nn.L1Loss()

    # to check if the optimizer can be printed as a string
    optimizer.__repr__()

    for i in range(5):
        optimizer.zero_grad()
        output = weight.mv(input) + bias
        loss = criterion(output, target)
        if i == 0:
            initial_value = loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    assert loss.item() < initial_value
    assert len(optimizer.fp32_param_groups) == len(optimizer.param_groups)

    for fp32_group, fp16_group in zip(optimizer.fp32_param_groups, optimizer.param_groups):
        for fp32_p, fp16_p in zip(fp32_group["params"], fp16_group["params"]):

            def assert_almost_zero(x):
                assert abs(x) < 1e-3
                return 1.0

            assert fp32_p.dtype == torch.float32
            if fp16_p.requires_grad:
                assert fp16_p.dtype == torch.float16
                (fp32_p - fp16_p).to("cpu").detach().apply_(assert_almost_zero)


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
def test_step_pure_fp16():
    weight = torch.randn(10, 5).half().cuda().requires_grad_()
    bias = torch.randn(10).half().cuda().requires_grad_()
    input = torch.randn(5).half().cuda()
    optimizer = Adam([weight, bias], lr=1e-3, optim_fp16=True)

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
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float16
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float16


@skip_if_no_cuda
@skip_if_no_adam
def test_step_pure_fp16_multigpu():
    if not torch.cuda.device_count() > 1:
        return
    weight = torch.randn(10, 5).half().cuda(0).requires_grad_()
    bias = torch.randn(10).half().cuda(1).requires_grad_()
    input = torch.randn(5).half().cuda(0)
    optimizer = Adam([weight, bias], lr=1e-3, optim_fp16=True)

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
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float16
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float16


@skip_if_no_cuda
@skip_if_no_adam
def test_pure_fp16_with_grad_scaler():
    weight = torch.randn(10, 5).cuda().half().requires_grad_()
    bias = torch.randn(10).cuda().half().requires_grad_()
    input = torch.randn(5).half().cuda()
    target = torch.randn(10).half().cuda()
    optimizer = Adam([weight, bias], lr=1e-3, optim_fp16=True)
    scaler = FairscaleGradScaler()
    criterion = torch.nn.L1Loss()

    # to check if the optimizer can be printed as a string
    optimizer.__repr__()

    for i in range(5):
        optimizer.zero_grad()
        output = weight.mv(input) + bias
        loss = criterion(output, target)
        if i == 0:
            initial_value = loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    assert loss.item() < initial_value
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float16
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float16


@skip_if_no_cuda
@skip_if_no_adam
def test_state_dict_pure_fp16():
    weight = torch.randn(10, 5).half().cuda().requires_grad_()
    bias = torch.randn(10).half().cuda().requires_grad_()
    input = torch.randn(5).half().cuda()

    optimizer = Adam([weight, bias], lr=1e-3, optim_fp16=True)

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
