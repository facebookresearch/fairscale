# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test AdaScale with a single node (1 CPU or 1 GPU). """

import tempfile

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import Linear, Sequential
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from fairscale.fair_dev.testing.golden_testing_data import adascale_test_data
from fairscale.fair_dev.testing.testing import make_cudnn_deterministic, skip_if_no_cuda
from fairscale.fair_dev.testing.testing_memory import find_tensor_by_shape
from fairscale.optim import AdaScale


def test_basic_cpu():
    """Test single batch behavior on CPU"""
    model = Linear(2, 2, bias=False)
    try:
        optim = AdaScale(SGD(model.parameters(), lr=0.1))
    except RuntimeError:
        return
    assert False, "Single batch AdaScale should not be suppported"


def test_loss_accum_cpu():
    """Test the loss accumulation behavior on CPU

    Loss accumulation is NOT SUPPORTED. This test shows that it does not work.
    """
    model = Linear(2, 2, bias=False)
    # num_gradients_to_accumulate value doesn't matter in this negative test.
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=3)
    # data 1
    in_data = Tensor([0.0, 1.0])
    loss = model(in_data).sum()
    # data 2
    in_data = Tensor([1.0, 0.0])
    loss += model(in_data).sum()
    # data 3
    in_data = Tensor([1.0, 2.0])
    loss += model(in_data).sum()
    # backward, but gradient is only produced once by the autograd engine.
    loss.backward()
    # The gain will always be 1, which renders adascale as noop.
    assert np.allclose(optim.gain(), 1.0), optim.gain()
    # We don't call optim.step(), since it will detect that backward is not yet done.


@pytest.mark.parametrize("cpu", [True, False])
@pytest.mark.parametrize("test_case", adascale_test_data)
@pytest.mark.parametrize("is_scaled_loss", [True, False])
def test_grad_accum(test_case, cpu, is_scaled_loss):
    """Test the basic functionality on CPU/GPU with gradient accumulation without DDP"""
    make_cudnn_deterministic()
    model = Linear(2, 2, bias=True)
    if not cpu:
        if torch.cuda.device_count() < 1:
            pytest.skip("1 GPU is required")
        model = model.cuda()
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2, is_scaled_loss=is_scaled_loss)
    expected_gain = test_case["expected_gain"]
    if "input" in test_case:
        data = [test_case["input"]] * 2
        gains = [expected_gain] * 2
    else:
        data = test_case["inputs"]
        gains = [None, expected_gain]
    for in_data, exp_gain in zip(data, gains):  # test 2 iterations catch more corner cases.
        # grad pass 1
        in_data_0 = Tensor(in_data[0])
        if not cpu:
            in_data_0 = in_data_0.cuda()
        loss = model(in_data_0).sum()
        if is_scaled_loss:
            loss = loss / 2
        loss.backward()
        # grad pass 2
        in_data_1 = Tensor(in_data[1])
        if not cpu:
            in_data_1 = in_data_1.cuda()
        loss = model(in_data_1).sum()
        if is_scaled_loss:
            loss = loss / 2
        loss.backward()

        if not is_scaled_loss:
            optim.scale_grad_by_num_grads_to_accum()

        if exp_gain is not None:
            assert np.allclose(optim.gain(), exp_gain), optim.gain()
            w, b = model.parameters()
            assert np.allclose(w.grad.cpu(), test_case["expected_grad"]), w.grad
            assert np.allclose(b.grad.cpu(), test_case["expected_bias_grad"]), b.grad
        # stepping it. Note that if we did more than 2 passes as promised by the
        # num_gradients_to_accumulate argument above, AdaScale is not be able to
        # detect that mistake for now. The result will just be wrong in that case.
        optim.step()
        optim.zero_grad()


@skip_if_no_cuda
def test_state_checkpointing():
    """Test state checkpointing on GPU since that's the common case.

    Note, we don't support checkpointing in the middle of gradient accumulation
    step. Therefore, it is not tested here.

    AdaScale doesn't have distributed state. Otherwise, it will need
    a unit test for checkpointing with DDP.
    """
    # Constants.
    num_grads_to_accum = 3
    in_dim = 5

    # Setup.
    def make_model_and_optim():
        model = Linear(in_dim, 2, bias=False)
        model = model.cuda()
        optim = AdaScale(SGD(model.parameters(), lr=0.1, momentum=0.9), num_gradients_to_accumulate=num_grads_to_accum)
        return model, optim

    model, optim = make_model_and_optim()

    # Run a bit.
    def run_a_bit(replay_data=None):
        data = []
        replay_data_idx = 0
        for _ in range(6):  # run some steps
            for i in range(num_grads_to_accum):
                if replay_data is None:
                    in_data = torch.rand(in_dim).cuda()
                    data.append(in_data)
                else:
                    in_data = replay_data[replay_data_idx]
                    replay_data_idx += 1
                out = model(in_data)
                out.sum().backward()
                if i == num_grads_to_accum - 1:
                    optim.step()
                    optim.zero_grad()
        return out, data

    run_a_bit()

    with tempfile.NamedTemporaryFile() as f:
        temp_file_name = f.name

        # Save a checkpoint.
        torch.save({"model": model.state_dict(), "optim": optim.state_dict()}, temp_file_name)

        # Train more.
        out, replay_data = run_a_bit()

        # Save the gain and out.
        expected_out = out.sum().item()
        expected_gain = optim.gain()

        # Load back the checkpoint.
        model, optim = make_model_and_optim()  # They both need to start afresh.
        ckpt = torch.load(temp_file_name)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])

        # Train the same steps.
        out, _ = run_a_bit(replay_data)

    # Assert the results.
    assert np.allclose(out.sum().item(), expected_out), out.sum().item()
    assert np.allclose(optim.gain(), expected_gain), optim.gain()


def test_lr_scheduler():
    """Test AdaScale working with torch.optim.lr_scheduler."""
    num_grads_to_accum = 3
    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=num_grads_to_accum)
    # We use 1, not 0.1 here since scheduler.step() is called here first.
    scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 / 10**epoch)
    for epoch in range(3):
        for data_idx in range(10):
            for accumulation in range(num_grads_to_accum):
                in_data = torch.rand(2)
                loss = model(in_data).sum()
                loss.backward()
            assert optim.gain() <= 3, optim.gain()
            optim.step()
            optim.zero_grad()
            # asserting LR is right
            assert np.allclose(optim.param_groups[0]["lr"], 0.1 / 10**epoch), optim.param_groups[0]["lr"]
        scheduler.step()
        # asserting LR is right
        assert np.allclose(optim.param_groups[0]["lr"], 0.1 / 10 ** (epoch + 1)), optim.param_groups[0]["lr"]


@skip_if_no_cuda
@pytest.mark.parametrize("debias_ewma", [True, False])
@pytest.mark.parametrize("is_scaled_loss", [True, False])
def test_add_param_group(debias_ewma, is_scaled_loss):
    """Test AdaScale supports add_param_group() API for both scaled and unscaled loss."""
    num_grads_to_accum = 2
    model1 = Linear(2, 2, bias=True)
    with torch.no_grad():
        # make weights and bias deterministic, which is needed for
        # multi-layer models. For them, adascale gain is affected by
        # parameters from other layers.
        model1.weight.copy_(Tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2))
        model1.bias.fill_(0.1)
    optim = AdaScale(
        SGD(model1.parameters(), lr=0.1),
        num_gradients_to_accumulate=2,
        is_scaled_loss=is_scaled_loss,
        debias_ewma=debias_ewma,
    )
    assert len(optim._hook_handles) == 2, len(optim._hook_handles)

    model2 = Linear(2, 3, bias=True)
    with torch.no_grad():
        # make weights and bias deterministic
        model2.weight.copy_(Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(3, 2))
        model2.bias.fill_(0.2)
    optim.add_param_group({"params": model2.parameters()})
    assert len(optim._hook_handles) == 4, len(optim._hook_handles)

    # make sure we can run the model.
    model = Sequential(model1, model2).cuda()
    in_data_0 = Tensor([1.0, 2.0]).cuda()
    loss = model(in_data_0).sum()
    if is_scaled_loss:
        loss = loss / num_grads_to_accum
    loss.backward()

    in_data_1 = Tensor([3.0, 4.0]).cuda()
    loss = model(in_data_1).sum()
    if is_scaled_loss:
        loss = loss / num_grads_to_accum
    loss.backward()

    if not is_scaled_loss:
        optim.scale_grad_by_num_grads_to_accum()

    # make sure the gains are right and we can step.
    # since this is the first step, debias_ewma doesn't affect the value.
    assert np.allclose(optim.gain(), 1.1440223454935758), optim.gain()
    assert np.allclose(optim.gain(0), 1.1428571428571428), optim.gain(0)
    assert np.allclose(optim.gain(1), 1.1471258476157762), optim.gain(1)
    optim.step()
    optim.zero_grad()

    # make sure we can add a PG again after stepping.
    model3 = Linear(3, 4, bias=True)
    with torch.no_grad():
        # make weights and bias deterministic
        model3.weight.copy_(Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2).reshape(4, 3))
        model3.bias.fill_(0.2)
    optim.add_param_group({"params": model3.parameters()})
    assert len(optim._hook_handles) == 6, len(optim._hook_handles)

    # make sure we can run the model.
    model = Sequential(model1, model2, model3).cuda()
    in_data_0 = Tensor([1.0, 2.0]).cuda()
    loss = model(in_data_0).sum()
    if is_scaled_loss:
        loss = loss / num_grads_to_accum
    loss.backward()

    in_data_1 = Tensor([3.0, 4.0]).cuda()
    loss = model(in_data_1).sum()
    if is_scaled_loss:
        loss = loss / num_grads_to_accum
    loss.backward()

    if not is_scaled_loss:
        optim.scale_grad_by_num_grads_to_accum()

    # make sure gains are right and we can step.
    # the last PG's gain is not affected by debias_ewma since it is the first step for that PG.
    assert np.allclose(optim.gain(), 1.1382937715383077 if debias_ewma else 1.1391959826562015), optim.gain()
    assert np.allclose(optim.gain(0), 1.142857206008338 if debias_ewma else 1.142857206006931), optim.gain(0)
    assert np.allclose(optim.gain(1), 1.1116875516387468 if debias_ewma else 1.1116906378271827), optim.gain(1)
    assert np.allclose(optim.gain(2), 1.0749164095196344), optim.gain(2)
    optim.step()
    optim.zero_grad()


@pytest.mark.parametrize(
    "test_case",
    [
        {"num_grads_to_accum": 3, "exp_gain": 2.141385737279438},
        {"num_grads_to_accum": 6, "exp_gain": 2.9927880097754036},
        {"num_grads_to_accum": 9, "exp_gain": 3.4461759591877312},
    ],
)
@pytest.mark.parametrize("is_scaled_loss", [True, False])
def test_set_num_gradients_to_accumulate(test_case, is_scaled_loss):
    """Test set_num_gradients_to_accumulate experimental feature."""
    num_grads_to_accum = test_case["num_grads_to_accum"]
    exp_gain = test_case["exp_gain"]

    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2, is_scaled_loss=is_scaled_loss)
    loss = model(Tensor([0.0, 1.0])).sum()
    if is_scaled_loss:
        loss = loss / 2
    loss.backward()
    loss = model(Tensor([1.0, 0.0])).sum()
    if is_scaled_loss:
        loss = loss / 2
    loss.backward()

    if not is_scaled_loss:
        optim.scale_grad_by_num_grads_to_accum()

    assert np.allclose(optim.gain(), 2.0)
    optim.step()
    optim.zero_grad()

    optim.set_scale(float(num_grads_to_accum))
    optim.set_num_gradients_to_accumulate(num_grads_to_accum)
    for _ in range(num_grads_to_accum):
        loss = model(Tensor([0.0, 1.0])).sum() / num_grads_to_accum
        if is_scaled_loss:
            loss = loss / num_grads_to_accum
        loss.backward()

    if not is_scaled_loss:
        optim.scale_grad_by_num_grads_to_accum()

    assert np.allclose(optim.gain(), exp_gain), optim.gain()
    optim.step()
    optim.zero_grad()


def test_debias_ewma():
    """Test debias_ewma experimental feature"""
    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2, debias_ewma=True)
    for _ in range(4):
        out = model(Tensor([0.0, 1.0]))
        out.sum().backward()
        out = model(Tensor([1.0, 0.0]))
        out.sum().backward()
        assert np.allclose(optim.gain(), 2.0), optim.gain()
        optim.step()
        optim.zero_grad()


def test_gradient_value():
    """Test that we don't mutate the gradients during backward"""
    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)

    # fwd 1
    out = model(Tensor([0.0, 1.0]))
    out.sum().backward()
    assert np.allclose(model.weight.grad.numpy(), [[0.0, 1.0], [0.0, 1.0]]), model.weight.grad

    # fwd 2, grad is accumulated
    out = model(Tensor([0.0, 1.0]))
    out.sum().backward()
    assert np.allclose(model.weight.grad.numpy(), [[0.0, 2.0], [0.0, 2.0]]), model.weight.grad

    # assert gain and grad value before/after step/zero_grad
    assert np.allclose(optim.gain(), 1.0000002499999376), optim.gain()
    optim.step()
    assert np.allclose(model.weight.grad.numpy(), [[0.0, 2.0], [0.0, 2.0]]), model.weight.grad
    optim.zero_grad()
    assert np.allclose(model.weight.grad.numpy(), [[0.0, 0.0], [0.0, 0.0]]), model.weight.grad


@pytest.mark.parametrize(
    "test_case",
    [
        {"scale": None, "exp_gain": 4.0},  # default, baseline is single batch
        {"scale": 4.0 / 3, "exp_gain": 4.0 / 3},  # baseline is grad_accum = 3
        {"scale": 4.0 / 2, "exp_gain": 2.0},  # baseline is grad_accum = 2
        {"scale": 4.0 / 1, "exp_gain": 4.0},  # baseline is single batch
    ],
)
def test_scale_not_equal_default(test_case):
    """Test gain value when scale doesn't equal world size * grad_accum"""
    scale = test_case["scale"]
    exp_gain = test_case["exp_gain"]
    model = Linear(4, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=4, scale=scale)

    data = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for i in range(4):
        out = model(Tensor(data[i]))
        out.sum().backward()
    # Since the inputs are perfect orthogonal, the gain should be at the scale.
    assert np.allclose(optim.gain(), exp_gain), optim.gain()


@skip_if_no_cuda
def test_unhook():
    """Test unhook that frees the tensor from CUDA memory."""
    model = Linear(123, 456, bias=False).cuda()  # unique shape so that it can be found
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)

    torch.cuda.empty_cache()
    target_shape = (456, 123)
    assert find_tensor_by_shape(target_shape), "something wrong with gc-based method to find the tensor"

    optim.unhook()
    del model
    del optim
    torch.cuda.empty_cache()
    assert not find_tensor_by_shape(target_shape), "tensor should have been released"


def test_custom_smoothing_factor():
    """Test custom smoothing since we had a bug around it."""
    model = Linear(1, 1)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), smoothing=0.12345, num_gradients_to_accumulate=3)
    assert optim._smoothing == 0.12345
