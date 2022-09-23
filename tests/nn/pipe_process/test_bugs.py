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
import torch.nn.functional as F

from fairscale.fair_dev.testing.testing import get_worker_map, torch_spawn
from fairscale.nn.pipe import AsyncPipe


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def python_autograd_function(pipe_class):
    # FIXME deadlock with AsyncPipe?
    # A Python autograd function might fail with this error:
    #
    #   RuntimeError: Returning Variables sharing storage with other Variables
    #   that require grad is not supported in Python functions. Please submit a
    #   feature request if you hit this error.
    #
    # It doesn't look like an essential restriction. But it happens on the
    # current PyTorch version. To avoid it, we should detach the tensor before
    # returning by identity autograd functions, such as Wait, Fork, and Join.

    torch.manual_seed(0)

    class Identity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad):
            return grad

    class M(nn.Module):
        def forward(self, input):
            return Identity.apply(input)

    model = nn.Sequential(M(), M())
    model = pipe_class(model, [1, 1], worker_map=get_worker_map(), checkpoint="always").cuda()
    model.eval()

    x = torch.rand(42)
    y = model(x)
    if model.group.rank() == 1:
        assert torch.allclose(x, y)

    torch.distributed.rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def exception_no_hang(pipe_class):
    # In v0.0.2, once a failed partition receives a normal message
    # (non-closing) for the next micro-batch, a hang occured. The reason was
    # that a failed partition didn't call in_queue.task_done() on a normal
    # message. So the former partition was blocked at out_queue.join() for the
    # next of next micro-batch.
    class ExpectedException(Exception):
        pass

    class Pass(nn.Module):
        def forward(self, x):
            return x

    class Raise(nn.Module):
        def forward(self, x):
            raise ExpectedException()

    model = nn.Sequential(Pass(), Pass(), Raise())
    model = pipe_class(model, [1, 1, 1], worker_map=get_worker_map(), chunks=3)
    model.eval()

    if model.group.rank() == 2:
        with pytest.raises(ExpectedException):
            model(torch.rand(3))
    else:
        model(torch.rand(3))

    torch.distributed.barrier()


@torch_spawn([2])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 cuda devices required")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def tuple_wait(cuda_sleep, pipe_class):
    # In v0.0.3, Wait is applied to only the first tensor on a micro-batch.
    # Under this behavior, if checkpointing was disabled, there's a possibility
    # that gradient accumulations on other tensors are not synchronized
    # properly to the copy stream.
    class Sleep(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.detach()

        @staticmethod
        def backward(ctx, grad):
            with torch.cuda.device(grad.device):
                cuda_sleep(0.05)
            return grad

    class Layer1(nn.Module):
        def forward(self, pair):
            a, b = pair
            return a * 1, b * 2, b * 3

    class Layer2(nn.Module):
        def forward(self, triple):
            a, b, c = triple
            b = Sleep.apply(b)
            return a + b + c

    model = nn.Sequential(Layer1(), Layer2())
    model = pipe_class(
        model,
        [1, 1],
        worker_map=get_worker_map(),
        input_device=torch.cuda.current_device(),
        chunks=32,
        checkpoint="never",
    ).cuda()

    a = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)
    b = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)

    y = model((a, b))
    if model.group.rank() == 1:
        y.norm().backward()
    else:
        model.back_helper(y)

    if model.group.rank() == 0:
        assert torch.isclose(b.grad.norm().cpu(), torch.tensor(5.000))


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def parallel_randoms(pipe_class):
    class Dropouts(nn.Module):
        def forward(self, x):
            for _ in range(100):
                x = F.dropout(x, p=0.001)
            return x

    model = nn.Sequential(Dropouts(), Dropouts())

    x = torch.rand(10, 10, requires_grad=True).cuda()
    x.retain_grad()
    model = pipe_class(
        model,
        [1, 1],
        input_device=torch.cuda.current_device(),
        worker_map=get_worker_map(),
        chunks=10,
        checkpoint="always",
    ).cuda()
    y = model(x)
    tensor_list = [torch.empty_like(x) for _ in range(2)]
    if model.group.rank() == 1:
        y.norm().backward()
        torch.distributed.barrier()
        tensor_list[model.group.rank()] = y
        torch.distributed.all_gather(tensor_list, y, group=model.group)
        assert tensor_list[0].to(torch.bool).tolist() == tensor_list[1].to(torch.bool).tolist()
    else:
        model.back_helper(y)
        torch.distributed.barrier()
        tensor_list[model.group.rank()] = x.grad
        torch.distributed.all_gather(tensor_list, x.grad, group=model.group)
