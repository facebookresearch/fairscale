# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import pytest
import torch
from torch import nn
import torch.cuda

from fairscale.nn.pipe.checkpoint import Checkpointing, Function, TensorOrTensors, is_checkpointing, is_recomputing
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


def make_checkpoint(function: Function, input: TensorOrTensors, index: int) -> TensorOrTensors:
    """Makes a checkpoint with a simple interface like
    :func:`torch.utils.checkpoint.checkpoint`. It's only used to test or debug
    :class:`Checkpoint` and :class:`Recompute` without boilerplate.
    """
    batch = Batch(input, index)

    chk = Checkpointing(function, batch)
    batch = chk.checkpoint()
    chk.recompute(batch)

    return batch.tensor_or_tensors


@pytest.mark.parametrize("device", devices)
def test_serial_checkpoints(device):
    # Copied from https://github.com/pytorch/pytorch/pull/18568.
    timeline = []

    class Log(torch.autograd.Function):
        @staticmethod
        def forward(ctx, name, x):
            ctx.name = name
            timeline.append(f"{name}:forward")
            return x.detach()

        @staticmethod
        def backward(ctx, grad_output):
            name = ctx.name
            timeline.append(f"{name}:backward")
            return None, grad_output

    a = torch.rand(1, device=device, requires_grad=True)
    b = torch.rand(1, device=device, requires_grad=True)

    # Increase the next function sequence number.
    _ = a + 1 + 2 + 3 + 4 + 5

    a = make_checkpoint(partial(Log.apply, "a"), a, 0)

    a, phony = fork(a)
    b = join(b, phony)

    b = make_checkpoint(partial(Log.apply, "b"), b, 0)

    c = torch.cat((a, b))

    out = c.sum()

    #                        +--> {a} --Checkpoint(Log)--> {a}
    # {out} --Sum--> {c} --Cat     ^-----------------------------+
    #                        +--> {b} --Checkpoint(Log)--> {b} --First--> {b}
    out.backward()

    assert timeline == ["a:forward", "b:forward", "b:forward", "b:backward", "a:forward", "a:backward"]
    #    |----------------------|  |-----------------------|  |-----------------------|
    #          forward pass            Checkpoint(Log[b])         Checkpoint(Log[a])


def test_not_requires_grad():
    x = Batch(torch.rand(1, requires_grad=False), 0)
    assert not x[0].requires_grad

    def f(x):
        return x * 2

    chk = Checkpointing(f, x)
    x = chk.checkpoint()
    assert x[0].requires_grad

    chk.recompute(x)
    assert x[0].requires_grad

    x.tensor.backward()


def test_not_requires_grad_with_parameter():
    x = torch.rand(1, requires_grad=False)
    a = torch.rand(1, requires_grad=True)

    def f(x):
        return x * a

    y = make_checkpoint(f, x, 0)
    y.backward()

    assert a.grad is not None


@pytest.mark.parametrize("device", devices)
def test_random_in_checkpoint(device):
    dropout = nn.Dropout(p=0.5)

    torch.manual_seed(0)
    x = torch.randn(3, 3, device=device, requires_grad=True)
    y = dropout(x)
    y.norm().backward()

    torch.manual_seed(0)
    chk_x = torch.randn(3, 3, device=device, requires_grad=True)
    chk_y = make_checkpoint(dropout, chk_x, 0)
    chk_y.norm().backward()

    assert torch.allclose(x.grad, chk_x.grad)


def test_detect_checkpointing_recomputing():
    logs = []

    class Detect(nn.Module):
        def forward(self, input):
            logs.append((is_checkpointing(), is_recomputing()))
            return input

    model = Detect()
    input = torch.rand(1, requires_grad=True)

    output = make_checkpoint(model, input, 0)
    output.backward()

    assert logs == [(True, False), (False, True)]


def test_detect_checkpointing_recomputing_without_checkpoint():
    logs = []

    class Detect(nn.Module):
        def forward(self, input):
            logs.append((is_checkpointing(), is_recomputing()))
            return input

    model = Detect()
    input = torch.rand(1, requires_grad=True)

    output = model(input)
    output.backward()

    assert logs == [(False, False)]


def test_non_grad_output():
    class ForkNonGrad(nn.Module):
        def forward(self, input):
            return (input * 2, torch.rand(1))

    model = ForkNonGrad()
    input = torch.rand(1, requires_grad=True)

    output = make_checkpoint(model, input, 0)
    output[0].backward()
