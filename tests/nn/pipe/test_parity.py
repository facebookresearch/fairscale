# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing Pipe Module Parity
"""

import contextlib
import copy

import numpy as np
import pytest
import torch

from fairscale.fair_dev.testing.testing import skip_if_single_gpu
from fairscale.nn import Pipe


def _get_model(num_inputs=2, num_hidden=20, num_outputs=2):
    num_layers = torch.cuda.device_count() - 2
    model = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, num_hidden),
        *([torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]),
        torch.nn.Linear(num_hidden, num_outputs),
    )
    return model


def _check_parity(rmodel, pmodel, ropt, popt, rloss, ploss):

    for pparams, rparams in zip(pmodel.parameters(), rmodel.parameters()):
        assert torch.allclose(pparams.cuda(), rparams, atol=1e-2), f"Model params are different {oparams} {rparams}"

    for p_pg, reg_pg in zip(popt.param_groups, ropt.param_groups):
        for p_pg, reg_pg in zip(p_pg["params"], reg_pg["params"]):
            assert torch.allclose(
                p_pg.cuda(), reg_pg, atol=1e-2
            ), f"Model parameters differ in between Pipe and Vanilla {[o_pg]} {reg_pg}"

        for p_buf, reg_buf in zip(pmodel.buffers(), rmodel.buffers()):
            assert torch.allclose(p_buf.cuda(), reg_buf, atol=1e-2), "Model buffers differ in between Pipe and Vanilla."


def _get_fp16_context(use_fp16=False):
    if use_fp16:
        return torch.cuda.amp.autocast()
    else:
        return contextlib.nullcontext()


def _train(model, optimizer, use_fp16):

    inputs = torch.ones(32, 2).cuda()
    labels = torch.ones(32, 2)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    model.train()
    with _get_fp16_context(use_fp16):
        pred = model(inputs)
        loss = loss_fn(pred, labels.to(pred.device))
        loss.backward()
    optimizer.step()
    return model, optimizer, loss


def _train_reg_model(model, use_fp16=False):
    model = copy.deepcopy(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return _train(model, optimizer, use_fp16)


def _train_pipe_model(model, use_fp16=False, checkpoint="never", chunks=1):
    model = copy.deepcopy(model)
    model = Pipe(
        model,
        balance=[1] * torch.cuda.device_count(),
        devices=list(range(torch.cuda.device_count())),
        chunks=chunks,
        checkpoint=checkpoint,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return _train(model, optimizer, use_fp16)


@skip_if_single_gpu
@pytest.mark.parametrize("use_fp16", [True, False])
@pytest.mark.parametrize("checkpoint", ["always", "except_last", "never"])
@pytest.mark.parametrize("chunks", [1, 4])
def test_correctness(use_fp16, checkpoint, chunks):
    torch.manual_seed(0)
    np.random.seed(0)

    if use_fp16 and not hasattr(torch.cuda.amp, "custom_fwd"):
        pytest.skip(f"AMP APIs are not supported in torch version {torch.__version__}")

    model = _get_model()
    rmodel, ropt, rloss = _train_reg_model(model)
    pmodel, popt, ploss = _train_pipe_model(
        model,
        use_fp16=use_fp16,
        checkpoint=checkpoint,
        chunks=chunks,
    )
    _check_parity(rmodel, pmodel, ropt, popt, rloss, ploss)
