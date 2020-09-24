# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# TODO (Min): enable mypy
# type: ignore


import numpy as np
import torch

import fairscale.optim.adascale as adascale


def test_object():
    params = [
        torch.tensor([[1.0, -1.0], [2.0, 3.0]], requires_grad=True),
        torch.tensor([[2.0, 3.0]], requires_grad=True),
    ]
    sgd = torch.optim.SGD(params, lr=0.1)
    obj = adascale.AdaScale(sgd, scale=1.0, num_replicas=1)
    assert obj._scale == 1.0
    obj._num_replicas = 8
    obj.set_scale(3.0)
    assert obj.scale == 3.0
    obj._num_replicas = 4
    obj.set_scale(3.0)
    assert obj.scale == 3.0
    assert obj.gain(2.0) == 1.0
    obj._state["var_avg"] = 3.0
    obj._state["norm_avg"] = 1.0
    assert obj.gain(3.0) == 2.0


def test_optimization_1():
    params_t = torch.Tensor([1.0, 1.5])

    params = torch.autograd.Variable(params_t, requires_grad=True)

    # See torch.test.test_optim
    # Also see Rosenbrock/banana function
    def rosenbrock(tensor):
        x, y = tensor
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    params_t = torch.Tensor([1.0, 1.5])

    params = torch.autograd.Variable(params_t, requires_grad=True)
    sgd = torch.optim.SGD([params], lr=0.001)
    schedule = torch.optim.lr_scheduler.MultiStepLR(sgd, [1000])
    obj = adascale.AdaScale(sgd, scale=2.0, num_replicas=1, patch_optimizer=True)
    i = 0.0
    while i < 100000:
        sgd.zero_grad()
        loss = rosenbrock(params)
        loss.backward()
        sgd.step()
        i += obj.gain(2.0)
        schedule.step()
    assert params.allclose(torch.tensor([1.0, 1.0]), atol=0.01)


def test_optimization_2():
    def rosenbrock_noisy(tensor):
        x, y = tensor
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 + np.random.normal(0.0, 10.0)

    params_t = torch.Tensor([1.0, 1.5])

    params = torch.autograd.Variable(params_t, requires_grad=True)
    sgd = torch.optim.SGD([params], lr=0.001)
    schedule = torch.optim.lr_scheduler.MultiStepLR(sgd, [1000])
    obj = adascale.AdaScale(sgd, scale=2.0, num_replicas=1, patch_optimizer=True)
    i = 0.0
    while i < 100000:
        sgd.zero_grad()
        loss = rosenbrock_noisy(params)
        loss.backward()
        sgd.step()
        i += obj.gain(2.0)
        schedule.step()
    assert params.allclose(torch.tensor([1.0, 1.0]), atol=0.01)
