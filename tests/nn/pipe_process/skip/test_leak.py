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

from fairscale.nn.pipe import Pipe, is_checkpointing, is_recomputing
from fairscale.nn.pipe.skip import pop, skippable, stash
from fairscale.nn.pipe.skip.tracker import current_skip_tracker
from tests.nn.model_parallel.commons import get_worker_map, torch_spawn


@skippable(stash=["skip"])
class Stash(nn.Module):
    def forward(self, input):
        yield stash("skip", input)
        return input


@skippable(pop=["skip"])
class Pop(nn.Module):
    def forward(self, input):
        skip = yield pop("skip")
        return input + skip


@torch_spawn([2])
@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("checkpoint", ["always", "except_last", "never"])
@pytest.mark.parametrize("pipeline_style", [Pipe.MultiProcess, Pipe.AsyncSchedule])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" in os.environ, reason="broken on mpi")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def delete_portal_tensor(train, checkpoint, pipeline_style):
    # Without checkpointing:
    # +- Stash --+  +--- Pop ----+ - - - layers
    # | 2,blue,1 |--| 1,orange,0 | - - - tensor_life and portal function
    # +----------+  +------------+
    #
    # With checkpointing:
    # +- Stash --+  +--- Pop ----+  +--- Pop'----+  +- Stash'--+
    # | 3,blue,2 |--| 2,orange,1 |--| 1,orange,0 |--| 1,blue,0 |
    # +----------+  +------------+  +------------+  +----------+

    if pipeline_style == Pipe.AsyncSchedule:
        pytest.skip("Skip tensors NYI for AsyncSchedule")

    def portal_tensor_life_is(tensor_life, skip_tracker=None):
        if skip_tracker is None:
            skip_tracker = current_skip_tracker()

        # Get the current portal.
        portal = list(skip_tracker.portals.values())[0]

        if tensor_life == 0:
            return portal.tensor_life == 0 and portal.tensor is None
        else:
            return portal.tensor_life == tensor_life and portal.tensor is not None

    # Check the portal tensor after 'Stash'.
    stash_ = Stash()

    @stash_.register_forward_hook
    def check_portal_tensor_after_stash(*_):
        if is_checkpointing():
            assert portal_tensor_life_is(2)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(1)

    pop_ = Pop()

    @pop_.register_forward_hook
    def check_portal_tensor_after_pop(*_):
        if is_checkpointing():
            assert portal_tensor_life_is(1)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(0)

    class NoPortalTensorAtBackward(nn.Module):
        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.skip_tracker = current_skip_tracker()
                return input.detach()

            @staticmethod
            def backward(ctx, grad):
                assert portal_tensor_life_is(0, skip_tracker=ctx.skip_tracker)
                return grad

        def forward(self, input):
            return self.F.apply(input)

    model = nn.Sequential(NoPortalTensorAtBackward(), stash_, pop_)
    model = Pipe(
        model, balance=[2, 1], style=pipeline_style, worker_map=get_worker_map(), chunks=2, checkpoint=checkpoint,
    )

    input = torch.rand(10, requires_grad=True)

    if train:
        model.train()
        output = model(input)
        if model.group.rank() == 1:
            output.norm().backward()
        else:
            model.back_helper(output)
    else:
        model.eval()
        with torch.no_grad():
            model(input)

    torch.distributed.barrier()
