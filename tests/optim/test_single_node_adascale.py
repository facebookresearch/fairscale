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
from torch.nn import Linear
from torch.optim import SGD

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
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=123)
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
    # therefore, the gain will always be 1, which renders adascale as noop.
    optim.step()
    assert np.allclose(optim.gain(), 1.0), optim.gain()


def test_grad_accum_cpu(cpu=True):
    """Test the basic functionality on CPU with gradient accumulation without DDP"""
    model = Linear(2, 2, bias=False)
    if not cpu:
        model = model.cuda()
    optim = AdaScale(SGD(model.parameters(), lr=0.1), num_gradients_to_accumulate=2)
    for expected_gain in [2.0, 2.0]:  # test 2 iterations catch more corner cases.
        # grad pass 1
        in_data = Tensor([0.0, 1.0])
        if not cpu:
            in_data = in_data.cuda()
        out = model(in_data)
        out.sum().backward()
        # grad pass 2
        in_data = Tensor([1.0, 0.0])
        if not cpu:
            in_data = in_data.cuda()
        out = model(in_data)
        out.sum().backward()
        # stepping it. Note that if we did more than 2 passes as promised by the
        # num_gradients_to_accumulate argument above, AdaScale is not be able to
        # detect that mistake for now. The result will just be wrong in that case.
        assert np.allclose(optim.gain(), expected_gain), optim.gain()
        optim.step()
        optim.zero_grad()


@skip_if_no_gpu
def test_grad_accum_gpu():
    """Test the basic functionality on GPU with gradient accumulation without DDP"""
    test_grad_accum_cpu(cpu=False)


@skip_if_no_gpu
def test_state_checkpointing():
    """ Test state checkpointing on GPU since that's the common case.

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
