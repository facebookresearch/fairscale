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

import os

import pytest
import torch
from torch import nn

from fairscale.nn.pipe import LazyModule, Pipe
from fairscale.nn.pipe.skip import pop, skippable, stash
from fairscale.nn.pipe.skip.portal import PortalBlue, PortalCopy, PortalOrange
from fairscale.utils.testing import get_worker_map, torch_spawn


@torch_spawn([3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("balance", [[3], [1, 2], [2, 1], [1, 1, 1]], ids=["3", "1:2", "2:1", "1:1:1"])
@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
@pytest.mark.parametrize("pipeline_style", [Pipe.MultiProcess, Pipe.AsyncSchedule])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" in os.environ, reason="broken on mpi")
def x1to3(balance, checkpoint, pipeline_style):
    torch.manual_seed(0)

    if pipeline_style == Pipe.AsyncSchedule and len(balance) > 1:
        print(f"skipping yarg")
        pytest.skip("Skip tensors NYI for AsyncSchedule")

    @skippable(stash=["1to3"])
    class Layer1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, input):
            yield stash("1to3", input)
            output = self.conv(input)
            return output

    class Layer2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, input):
            output = self.conv(input)
            return output

    @skippable(pop=["1to3"])
    class Layer3(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, input):
            skip_1to3 = yield pop("1to3")
            output = self.conv(input) + skip_1to3
            return output

    model = nn.Sequential(Layer1(), Layer2(), Layer3())
    model = Pipe(
        model,
        balance,
        chunks=3,
        checkpoint=checkpoint,
        input_device=torch.cuda.current_device(),
        style=pipeline_style,
        worker_map=get_worker_map(),
        pipelined_backward=False,
    ).cuda()

    input = torch.rand(30, 3, 224, 224, requires_grad=True).cuda()
    input.retain_grad()
    output = model(input)
    if model.group.rank() == len(balance) - 1:
        loss = output.mean()
        loss.backward()
    elif model.group.rank() < len(balance) - 1:
        model.back_helper(output)
    if model.group.rank() == len(balance) - 1:
        # TODO(tom) the single-process test uses 2e-1 but for some reason
        # mutli-process is more noisy, need to investigate why
        assert torch.allclose(output.norm(), torch.tensor(1039.0).cuda(), atol=4e-1)
    if model.group.rank() == 0:
        assert torch.allclose(input.grad.norm(), torch.tensor(0.0004533053).cuda())

    torch.distributed.barrier()


@torch_spawn([2])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" in os.environ, reason="broken on mpi")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("pipeline_style", [Pipe.MultiProcess, Pipe.AsyncSchedule])
def none_skip(pipeline_style):
    if pipeline_style == Pipe.AsyncSchedule:
        pytest.skip("Skip tensors NYI for AsyncSchedule")

    @skippable(stash=["none"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("none", None)
            return input

    @skippable(pop=["none"])
    class Pop(nn.Module):
        def forward(self, input):
            none = yield pop("none")
            assert none is None
            return input

    model = nn.Sequential(Stash(), Pop())
    model = Pipe(
        model,
        [1, 1],
        style=pipeline_style,
        worker_map=get_worker_map(),
        input_device=torch.cuda.current_device(),
        chunks=5,
    ).cuda()

    input = torch.rand(10, requires_grad=True).cuda()
    input.retain_grad()
    output = model(input)

    def assert_grad_fn_is_not_portal(grad_fn, visited=set()):
        if grad_fn in visited or grad_fn is None:
            return

        assert not isinstance(grad_fn, PortalBlue._backward_cls)
        assert not isinstance(grad_fn, PortalCopy._backward_cls)
        assert not isinstance(grad_fn, PortalOrange._backward_cls)

        visited.add(grad_fn)
        for next_grad_fn, _ in grad_fn.next_functions:
            assert_grad_fn_is_not_portal(next_grad_fn, visited)

    if model.group.rank() == 1:
        assert_grad_fn_is_not_portal(output.grad_fn)

        output.sum().backward()
    else:
        model.back_helper(output)
        assert input.grad.mean().item() == 1


@torch_spawn([2])
@pytest.mark.parametrize("pipeline_style", [Pipe.MultiProcess, Pipe.AsyncSchedule])
def lazy_skippable_error(pipeline_style):
    """Using skippable layers in combination with lazy construction is currently
    not supported, check that it raises an Exception"""

    @skippable(stash=["1to3"])
    class Layer1(nn.Linear):
        pass

    @skippable(pop=["1to3"])
    class Layer3(nn.Linear):
        pass

    model = [
        LazyModule(lambda: Layer1(10, 10)),
        LazyModule(lambda: nn.Linear(10, 10)),
        LazyModule(lambda: Layer3(10, 10)),
    ]

    with pytest.raises(ValueError, match="Can't use Skippable layers with multi-process pipe and lazy construction"):
        Pipe(
            model, [2, 1], style=pipeline_style, worker_map=get_worker_map(),
        )
