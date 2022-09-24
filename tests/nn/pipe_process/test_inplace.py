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

import pytest
import torch
from torch import nn

from fairscale.fair_dev.testing.testing import get_worker_map, torch_spawn
from fairscale.nn.pipe import AsyncPipe


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def inplace_on_requires_grad(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(inplace=True))
    model = pipe_class(model, [1, 1], worker_map=get_worker_map(), checkpoint="always")

    x = torch.rand(1)

    if pipe_class == AsyncPipe and model.group.rank() == 0:
        # With AsyncPipe, model will wait forever for gradients if not eval
        model.eval()

    y = model(x)

    message = r"a leaf Variable that requires grad .* used in an in-place operation."
    if model.group.rank() == 1:
        with pytest.raises(RuntimeError, match=message):
            y.backward()

    torch.distributed.barrier()


@torch_spawn([1])
@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def inplace_on_not_requires_grad(pipe_class):
    # In-place operation on a tensor not requiring grad doesn't cause a
    # RuntimeError. Currently, we cannot detect this case.
    model = nn.Sequential(nn.ReLU(inplace=True))
    model = pipe_class(model, [1], worker_map=get_worker_map(), checkpoint="always")

    x = torch.rand(1)
    y = model(x)
    del model

    message = r"a leaf Variable that requires grad .* used in an in-place operation."
    with pytest.raises(RuntimeError, match=message):
        y.backward()

    torch.distributed.barrier()


@torch_spawn([1])
@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def inplace_incorrect_grad(pipe_class):
    class M(nn.Module):
        def forward(self, foo_bar):
            # 'foo' requires grad but 'bar' does not. In-place operation on
            # 'bar' won't cause a RuntimeError.
            foo, bar = foo_bar

            # add_(1) is not idempotent, in contrast to relu_(). If it is
            # executed multiple times, it will accumulates each difference onto
            # 'bar'.
            bar.add_(1)

            # 'bar' is still captured by checkpointing. 'foo' will get
            # incorrect grad.
            return foo * bar

    model = nn.Sequential(M())
    model = pipe_class(model, [1], worker_map=get_worker_map(), checkpoint="always")

    foo = torch.tensor([1.0], requires_grad=True)
    bar = torch.tensor([1.0])

    output = model((foo, bar))
    del model
    output.backward()

    # The gradient of 'foo' should be 2, but it is 3 actually because
    # bar.add_(1) was executed twice due to checkpointing.
    assert foo.grad.item() == 2.0
