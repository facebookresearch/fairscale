# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing Offload Module
"""

import copy

import numpy as np
import torch

from fairscale.nn.misc.offload import OffloadModel


def _init():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    offload_device = torch.device("cpu")
    return device, offload_device


def test_single_run():
    device, offload_device = _init()
    model = _get_model()

    offload_model = OffloadModel(model_cpu=model, device=device, offload_device=offload_device, n_slices=2,)
    offload_optimizer = torch.optim.SGD(offload_model.parameters(), lr=0.001)

    input = torch.ones(2, 2).to(device)
    labels = torch.ones(2, 2).to(device)
    offload_model.train()
    pred = offload_model(input)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    loss = loss_fn(pred, labels)
    loss.backward()
    offload_optimizer.step()


def _get_model(num_inputs=2, num_hidden=2, num_layers=1, num_outputs=2):
    model = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, num_hidden),
        *([torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]),
        torch.nn.Linear(num_hidden, num_outputs),
    ).cuda()
    return model


def _check_parity(rmodel, omodel, ropt, oopt, rloss, oloss):

    for oparams, rparams in zip(omodel.parameters(), rmodel.parameters()):
        assert torch.allclose(oparams, rparams, atol=1e-2), f"Model params are different {oparams} {rparams}"

    for o_pg, reg_pg in zip(oopt.param_groups, ropt.param_groups):
        for o_pg, reg_pg in zip(o_pg["params"], reg_pg["params"]):
            assert torch.allclose(
                o_pg, reg_pg, atol=1e-2
            ), f"Model parameters differ in between Offlad and Vanilla {[o_pg]} {reg_pg}"

        for o_buf, reg_buf in zip(omodel.buffers(), rmodel.buffers()):
            assert torch.allclose(o_buf, reg_buf, atol=1e-2), "Model buffers differ in between Offload and Vanilla."


def test_correctness():
    device, offload_device = _init()
    model = _get_model().cuda()

    def train(model, optimizer):

        input = torch.ones(2, 2).to(device)
        labels = torch.ones(2, 2).to(device)
        model.train()
        pred = model(input)
        loss_fn = torch.nn.MSELoss(reduction="sum")
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        return model, optimizer, loss

    def train_reg_model():
        reg_model = copy.deepcopy(model)
        reg_optimizer = torch.optim.SGD(reg_model.parameters(), lr=0.001)
        return train(reg_model, reg_optimizer)

    def train_offload_model():
        omodel = copy.deepcopy(model)
        offload_model = OffloadModel(model_cpu=omodel, device=device, offload_device=offload_device, n_slices=2,)
        offload_optimizer = torch.optim.SGD(offload_model.parameters(), lr=0.001)
        return train(offload_model, offload_optimizer)

    rmodel, ropt, rloss = train_reg_model()
    omodel, oopt, oloss = train_offload_model()
    _check_parity(rmodel.cpu(), omodel.cpu(), ropt, oopt, rloss, oloss)
