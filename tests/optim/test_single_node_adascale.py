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

from fairscale.optim import AdaScale

skip_if_no_gpu = pytest.mark.skipif(torch.cuda.device_count() < 1, reason="1 GPU is required")


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


# IMPORTANT: make sure these test_cases values are sync'ed with the DDP
# test in test_ddp_adascale.py. This way, we make sure gradient accumulation
# works exactly like that in DDP.
@pytest.mark.parametrize("cpu", [True, False])
@pytest.mark.parametrize(
    "test_case",
    [
        # "input" value is a list of input tensors for micro-batch 0 and micro-batch 1.
        {"input": [[1.0, 0], [0, 1.0]], "expected_gain": 2.0},
        {"input": [[1.0, 1.0], [1.0, 1.0]], "expected_gain": 1.0000001249999846},
        {"input": [[-1.0, 1.0], [1.0, -1.0]], "expected_gain": 2.0},
        {"input": [[1.0, 4.0], [5.0, 0.5]], "expected_gain": 1.5022222222222221},
        {"input": [[-0.2, 3.0], [5.0, 0.5]], "expected_gain": 1.9433267229211089},
        # "inputs" to trigger multiple iteration tests, which make sure the
        # smoothing factor calculation is also covered.
        {"inputs": [[[-0.2, 3.3], [5.2, 0.7]], [[1.0, 4.0], [3.1, 0.1]]], "expected_gain": 1.744159431359284},
    ],
)
def test_grad_accum(test_case, cpu):
    """Test the basic functionality on CPU/GPU with gradient accumulation without DDP"""
    model = Linear(2, 2, bias=False)
    if not cpu:
        if torch.cuda.device_count() < 1:
            pytest.skip("1 GPU is required")
        model = model.cuda()
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)
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
        out = model(in_data_0)
        out.sum().backward()
        # grad pass 2
        in_data_1 = Tensor(in_data[1])
        if not cpu:
            in_data_1 = in_data_1.cuda()
        out = model(in_data_1)
        out.sum().backward()
        if exp_gain is not None:
            assert np.allclose(optim.gain(), exp_gain), optim.gain()
        # stepping it. Note that if we did more than 2 passes as promised by the
        # num_gradients_to_accumulate argument above, AdaScale is not be able to
        # detect that mistake for now. The result will just be wrong in that case.
        optim.step()
        optim.zero_grad()


@skip_if_no_gpu
def test_state_checkpointing():
    """ Test state checkpointing on GPU since that's the common case.

        Note, we don't support checkpointing in the middle of gradient accumulation
        step. Therefore, it is not tested here.

        AdaScale doesn't have distributed state. Otherwise, it will need
        a unit test for checkpointing with DDP.
    """
    # Constants.
    accum_steps = 3
    in_dim = 5

    # Setup.
    def make_model_and_optim():
        model = Linear(in_dim, 2, bias=False)
        model = model.cuda()
        optim = AdaScale(SGD(model.parameters(), lr=0.1, momentum=0.9), num_gradients_to_accumulate=accum_steps)
        return model, optim

    model, optim = make_model_and_optim()

    # Run a bit.
    def run_a_bit(replay_data=None):
        print("running")
        data = []
        replay_data_idx = 0
        for _ in range(6):  # run some steps
            for i in range(accum_steps):
                if replay_data is None:
                    in_data = torch.rand(in_dim).cuda()
                    data.append(in_data)
                else:
                    in_data = replay_data[replay_data_idx]
                    replay_data_idx += 1
                out = model(in_data)
                out.sum().backward()
                # print(out.sum().item())
                print(model.weight.grad)
                if i == accum_steps - 1:
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
    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=3)
    # We use 1, not 0.1 here since scheduler.step() is called here first.
    scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 / 10 ** epoch)
    for epoch in range(3):
        for data_idx in range(10):
            for accumulation in range(3):
                in_data = torch.rand(2)
                loss = model(in_data).sum()
                loss.backward()
            assert optim.gain() <= 3, optim.gain()
            optim.step()
            # asserting LR is right
            assert np.allclose(optim.param_groups[0]["lr"], 0.1 / 10 ** epoch), optim.param_groups[0]["lr"]
        scheduler.step()
        # asserting LR is right
        assert np.allclose(optim.param_groups[0]["lr"], 0.1 / 10 ** (epoch + 1)), optim.param_groups[0]["lr"]


@skip_if_no_gpu
@pytest.mark.parametrize("debias_ewma", [True, False])
def test_add_param_group(debias_ewma):
    """Test AdaScale supports add_param_group() API."""
    model1 = Linear(2, 2, bias=True)
    with torch.no_grad():
        # make weights and bias deterministic, which is needed for
        # multi-layer models. For them, adascale gain is affected by
        # parameters from other layers.
        model1.weight.copy_(Tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2))
        model1.bias.fill_(0.1)
    optim = AdaScale(SGD(model1.parameters(), lr=0.1), num_gradients_to_accumulate=2, debias_ewma=debias_ewma)
    assert len(optim._hook_handles) == 2

    model2 = Linear(2, 3, bias=True)
    with torch.no_grad():
        # make weights and bias deterministic
        model2.weight.copy_(Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(3, 2))
        model2.bias.fill_(0.2)
    optim.add_param_group({"params": model2.parameters()})
    assert len(optim._hook_handles) == 4

    # make sure we can run the model.
    model = Sequential(model1, model2).cuda()
    in_data_0 = Tensor([1.0, 2.0]).cuda()
    out = model(in_data_0)
    out.sum().backward()

    in_data_1 = Tensor([3.0, 4.0]).cuda()
    out = model(in_data_1)
    out.sum().backward()

    # make sure the gains are right and we can step.
    # since this is the first step, debias_ewma doesn't affect the value.
    assert np.allclose(optim.gain(), 1.1440223454935758)
    assert np.allclose(optim.gain(0), 1.1428571428571428)
    assert np.allclose(optim.gain(1), 1.1471258476157762)
    optim.step()

    # make sure we can add a PG again after stepping.
    model3 = Linear(3, 4, bias=True)
    with torch.no_grad():
        # make weights and bias deterministic
        model3.weight.copy_(Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2).reshape(4, 3))
        model3.bias.fill_(0.2)
    optim.add_param_group({"params": model3.parameters()})
    assert len(optim._hook_handles) == 6

    # make sure we can run the model.
    model = Sequential(model1, model2, model3).cuda()
    in_data_0 = Tensor([1.0, 2.0]).cuda()
    out = model(in_data_0)
    out.sum().backward()

    in_data_1 = Tensor([3.0, 4.0]).cuda()
    out = model(in_data_1)
    out.sum().backward()

    # make sure gains are right and we can step.
    # the last PG's gain is not affected by debias_ewma since it is the first step for that PG.
    assert np.allclose(optim.gain(), 1.1711342960340743 if debias_ewma else 1.1760960226735786)
    assert np.allclose(optim.gain(0), 1.0045687695285042 if debias_ewma else 1.0045776319944453)
    assert np.allclose(optim.gain(1), 1.2184881264548717 if debias_ewma else 1.2184877742714733)
    assert np.allclose(optim.gain(2), 1.117381091722702)
    optim.step()


@pytest.mark.parametrize(
    "test_case",
    [
        {"new_accum": 3, "exp_gain": 1.1190662277452972},
        {"new_accum": 6, "exp_gain": 1.0619749887486511},
        {"new_accum": 9, "exp_gain": 1.0339000994448166},
    ],
)
def test_set_num_gradients_to_accumulate(test_case):
    """Test set_num_gradients_to_accumulate experimental feature."""
    new_accum = test_case["new_accum"]
    exp_gain = test_case["exp_gain"]

    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)
    out = model(Tensor([0.0, 1.0]))
    out.sum().backward()
    out = model(Tensor([1.0, 0.0]))
    out.sum().backward()
    assert np.allclose(optim.gain(), 2.0)
    optim.step()

    optim.set_scale(float(new_accum))
    optim.set_num_gradients_to_accumulate(new_accum)
    for _ in range(new_accum):
        out = model(Tensor([0.0, 1.0]))
        out.sum().backward()

    assert np.allclose(optim.gain(), exp_gain), optim.gain()
    optim.step()


def test_debias_ewma():
    """Test debias_ewma experimental feature"""
    model = Linear(2, 2, bias=False)
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2, debias_ewma=True)
    for _ in range(4):
        optim.zero_grad()
        out = model(Tensor([0.0, 1.0]))
        out.sum().backward()
        out = model(Tensor([1.0, 0.0]))
        out.sum().backward()
        assert np.allclose(optim.gain(), 2.0), optim.gain()
        optim.step()
