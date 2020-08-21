# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import functools

import pytest
import torch

try:
    from fairscale.optim.adam import Adam, Precision

    imported_adam = True
except ImportError:
    imported_adam = False

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_no_adam = pytest.mark.skipif(not imported_adam, reason="Fairscale Adam not available")


@pytest.fixture(autouse=True)
def set_torch_seed():
    torch.manual_seed(1)
    yield


def make_full_precision_params():
    weight = torch.randn(2, 1).cuda().requires_grad_()
    bias = torch.randn(2).cuda().requires_grad_()
    input = torch.randn(1).cuda()

    return weight, bias, input


def make_half_precision_params():
    weight = torch.randn(2, 1).cuda().half().requires_grad_()
    bias = torch.randn(2).cuda().half().requires_grad_()
    input = torch.randn(1).half().cuda()

    return weight, bias, input


def step_test(optimizer, weight, bias, input):
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


def state_dict_test(optimizer, weight, bias, input):
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
    optimizer_c = Adam([weight_c, bias_c], lr=1e-3, precision=optimizer.precision)
    fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c, input)
    # Load state dict
    state_dict = deepcopy(optimizer.state_dict())
    optimizer_c.load_state_dict(state_dict)
    # Run both optimizations in parallel
    for _i in range(5):
        optimizer.step(fn)
        optimizer_c.step(fn_c)

        # TODO: Optimizer state gets cast to FP16 and back to FP32 for 
        # mixed-precision and memory-efficient mixed-precision, resulting
        # in a loss of precision, so we don't expect the parameters to
        # remain the exact same. This therefore does NOT pass for all
        # torch manual seeds
        (weight - weight_c).to("cpu").detach().apply_(assert_almost_zero)
        (bias - bias_c).to("cpu").detach().apply_(assert_almost_zero)


def assert_almost_zero(x):
    assert abs(x) < 1e-3
    return 1.0


@skip_if_no_cuda
@skip_if_no_adam
def test_step_full_precision_inferred():
    weight, bias, input = make_full_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3)

    step_test(optimizer, weight, bias, input)

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                assert p.dtype == torch.float32
    assert not optimizer.fp32_param_groups

    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float32
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float32


@skip_if_no_cuda
@skip_if_no_adam
def test_step_mixed_precision_inferred():
    weight, bias, input = make_half_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3)
    step_test(optimizer, weight, bias, input)

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

    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float32
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float32


@skip_if_no_cuda
@skip_if_no_adam
def test_step_memory_efficient():
    weight, bias, input = make_half_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3, precision=Precision.MEMORY_EFFICIENT_MIXED_PRECISION)
    step_test(optimizer, weight, bias, input)

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                assert p.dtype == torch.float16

    assert not optimizer.fp32_param_groups

    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float32
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float32


@skip_if_no_cuda
@skip_if_no_adam
def test_step_pure_fp16():
    weight, bias, input = make_half_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3, precision=Precision.PURE_FP16)
    step_test(optimizer, weight, bias, input)

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                assert p.dtype == torch.float16

    assert optimizer.state[weight]["exp_avg"].dtype == torch.float16
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float16

    assert not optimizer.fp32_param_groups


@skip_if_no_cuda
@skip_if_no_adam
def test_step_multigpu():
    if not torch.cuda.device_count() > 1:
        return
    weight = torch.randn(10, 5).cuda(0).requires_grad_()
    bias = torch.randn(10).cuda(1).requires_grad_()
    input = torch.randn(5).cuda(0)
    optimizer = Adam([weight, bias], lr=1e-3)

    step_test(optimizer, weight, bias, input)


@skip_if_no_cuda
@skip_if_no_adam
def test_step_multigpu_mixed_precision():
    if not torch.cuda.device_count() > 1:
        return
    weight = torch.randn(10, 5).cuda(0).half().requires_grad_()
    bias = torch.randn(10).cuda(1).half().requires_grad_()
    input = torch.randn(5).cuda(0).half()
    optimizer = Adam([weight, bias], lr=1e-3)

    step_test(optimizer, weight, bias, input)


@skip_if_no_cuda
@skip_if_no_adam
def test_step_pure_fp16_multigpu():
    if not torch.cuda.device_count() > 1:
        return
    weight = torch.randn(10, 5).half().cuda(0).requires_grad_()
    bias = torch.randn(10).half().cuda(1).requires_grad_()
    input = torch.randn(5).half().cuda(0)
    optimizer = Adam([weight, bias], lr=1e-3, precision=Precision.PURE_FP16)

    step_test(optimizer, weight, bias, input)

    assert optimizer.state[weight]["exp_avg"].dtype == torch.float16
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg"].dtype == torch.float16
    assert optimizer.state[bias]["exp_avg_sq"].dtype == torch.float16


@skip_if_no_cuda
@skip_if_no_adam
def test_state_dict_full_precision():
    weight, bias, input = make_full_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3)

    state_dict_test(optimizer, weight, bias, input)


@skip_if_no_cuda
@skip_if_no_adam
def test_state_dict_mixed_precision():
    weight, bias, input = make_half_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3, precision=Precision.MIXED_PRECISION)

    state_dict_test(optimizer, weight, bias, input)


@skip_if_no_cuda
@skip_if_no_adam
def test_state_dict_memory_efficient():
    weight, bias, input = make_half_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3, precision=Precision.MEMORY_EFFICIENT_MIXED_PRECISION)

    state_dict_test(optimizer, weight, bias, input)


@skip_if_no_cuda
@skip_if_no_adam
def test_state_dict_pure_fp16():
    weight, bias, input = make_half_precision_params()
    optimizer = Adam([weight, bias], lr=1e-3, precision=Precision.PURE_FP16)

    state_dict_test(optimizer, weight, bias, input)


@skip_if_no_cuda
@skip_if_no_adam
def test_build_fp32_params():
    weight = torch.randn(10, 5).cuda().half().requires_grad_()
    bias = torch.randn(10).cuda().half().requires_grad_()
    optimizer = Adam([weight, bias], lr=1e-3)
    optimizer._build_fp32_params([weight, bias])
    for fp32_group, fp16_group in zip(optimizer.fp32_param_groups, optimizer.param_groups):
        for fp32_p, fp16_p in zip(fp32_group["params"], fp16_group["params"]):
            assert fp32_p.dtype == torch.float32
            if fp16_p.requires_grad:
                assert fp16_p.dtype == torch.float16
                (fp32_p - fp16_p).to("cpu").detach().apply_(assert_almost_zero)


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


@skip_if_no_cuda
@skip_if_no_adam
def test_mixed_precision_with_full_precision_parameters():
    weight = torch.randn(10, 5, requires_grad=True).float().cuda()
    bias = torch.randn(10, requires_grad=True).float().cuda()
    with pytest.raises(AssertionError):
        Adam([weight, bias], lr=1e-2, precision=Precision.MIXED_PRECISION)


@skip_if_no_cuda
@skip_if_no_adam
def test_memory_efficient_with_full_precision_parameters():
    weight = torch.randn(10, 5, requires_grad=True).float().cuda()
    bias = torch.randn(10, requires_grad=True).float().cuda()
    with pytest.raises(AssertionError):
        Adam([weight, bias], lr=1e-2, precision=Precision.MEMORY_EFFICIENT_MIXED_PRECISION)


@skip_if_no_cuda
@skip_if_no_adam
def test_pure_fp16_with_full_precision_parameters():
    weight = torch.randn(10, 5, requires_grad=True).float().cuda()
    bias = torch.randn(10, requires_grad=True).float().cuda()
    with pytest.raises(AssertionError):
        Adam([weight, bias], lr=1e-2, precision=Precision.PURE_FP16)
