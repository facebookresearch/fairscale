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

from collections import OrderedDict
from copy import deepcopy
import os
import time

import pytest
import torch
from torch import nn

from fairscale.fair_dev.testing.testing import get_worker_map, torch_spawn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from fairscale.nn.pipe import AsyncPipe
from fairscale.nn.pipe.types import LazyModule


@torch_spawn([2])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def parameters(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))
    pipe = pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=1)
    if torch.distributed.get_rank() == 0:
        assert list(pipe.parameters()) != []
    else:
        assert list(pipe.parameters()) == []


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def infiniband():
    if torch.distributed.get_rank() == 0:
        t = torch.Tensor(range(100)).cuda()
        torch.distributed.broadcast(t, 0)
    else:
        t = torch.empty(100).cuda()
        torch.distributed.broadcast(t, 0)

    assert torch.equal(t, torch.Tensor(range(100)).cuda())
    print(f"t on {torch.distributed.get_rank()} is {t}")


@torch_spawn([2])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def infiniband2():
    if torch.distributed.get_rank() == 0:
        t = torch.Tensor(range(100)).cuda()
        torch.distributed.send(t, 1, group=get_pipeline_parallel_group())
    else:
        t = torch.empty(100).cuda()
        torch.distributed.recv(t, 0, group=get_pipeline_parallel_group())

    assert torch.equal(t, torch.Tensor(range(100)).cuda())
    print(f"t on {torch.distributed.get_rank()} is {t}")


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def infiniband3():
    t = torch.Tensor(range(100)).cuda()
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    assert torch.equal(t, torch.Tensor(range(0, 200, 2)).cuda())


@torch_spawn([2])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def mpi():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.distributed.barrier()
    tensor_size = (1024, 1024, 10)
    torch.cuda.set_device(torch.distributed.get_rank())  # need to pin device or ucx gets unhappy

    if torch.distributed.get_rank() == 0:
        # t = torch.Tensor(range(10)).cuda(0)
        t = torch.rand(*tensor_size).cuda(0)
        torch.distributed.send(t, 1, tag=1234)
    else:
        t = torch.empty(*tensor_size).cuda(1)
        torch.distributed.recv(t, 0, tag=1234)
        t2 = torch.rand(*tensor_size).cuda(1)

        assert torch.equal(t, t2)


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def public_attrs(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))

    pipe = pipe_class(
        model,
        balance=(1,),
        worker_map=get_worker_map(),
        chunks=42,
        checkpoint="always",
    )

    assert pipe.balance == [1]
    assert pipe.chunks == 42
    assert isinstance(pipe.chunks, int)
    assert pipe.checkpoint == "always"
    assert isinstance(pipe.checkpoint, str)


@torch_spawn([2])
@pytest.mark.parametrize("balance", [[2], [1, 1]])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def sequential_like(balance, pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = pipe_class(model, balance, worker_map=get_worker_map())

    if balance == [2]:
        if torch.distributed.get_rank() == 0:
            assert len(model) == 2
            assert list(model) == [a, b]

            assert model[0] is a
            assert model[1] is b
            with pytest.raises(IndexError):
                _ = model[2]

            assert model[-1] is b
            assert model[-2] is a
        else:
            assert len(model) == 0
            assert list(model) == []
    else:
        assert len(model) == 1
        if torch.distributed.get_rank() == 0:
            assert list(model) == [a]
            assert model[0] is a
            assert model[-1] is a
        else:
            assert list(model) == [b]
            assert model[0] is b
            assert model[-1] is b

        with pytest.raises(IndexError):
            _ = model[1]


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def balance_wrong_length(pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        pipe_class(model, balance=[1], worker_map=get_worker_map())

    with pytest.raises(ValueError):
        pipe_class(model, balance=[3], worker_map=get_worker_map())


@torch_spawn([2])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def balance_less_than_1(pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        pipe_class(model, balance=[0, 2], worker_map=get_worker_map())

    with pytest.raises(ValueError):
        pipe_class(model, balance=[-1, 3], worker_map=get_worker_map())


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def chunks_less_than_1(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError):
        pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=0)

    with pytest.raises(ValueError):
        pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=-1)


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def too_few_devices(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))

    with pytest.raises(IndexError):
        # len(balance) > len(group.size())
        model = pipe_class(model, balance=[1, 1, 1, 1], worker_map=get_worker_map())


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def batch_size_indivisible(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))
    model = pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(7, 1))

    # Indivisible batch size is legal.
    assert not record


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def batch_size_small(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))
    model = pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(2, 1))

    # Batch size smaller than chunks is legal.
    assert not record


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def checkpoint_mode(pipe_class):
    def count_grad_fn(grad_fn, name, visited=set()):
        if grad_fn in visited:
            return 0
        visited.add(grad_fn)

        if grad_fn is None:
            return 0
        if grad_fn.__class__.__name__ == name:
            return 1

        counter = 0
        for next_grad_fn, _ in grad_fn.next_functions:
            counter += count_grad_fn(next_grad_fn, name, visited=visited)
        return counter

    model = nn.Sequential(nn.Linear(1, 1))
    input = torch.rand(2, 1)

    always = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint="always",
    )
    except_last = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint="except_last",
    )
    never = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint="never",
    )

    always_output = always(input)
    except_last_output = except_last(input)
    never_output = never(input)

    assert count_grad_fn(always_output.grad_fn, "CheckpointBackward") == 2
    assert count_grad_fn(except_last_output.grad_fn, "CheckpointBackward") == 1
    assert count_grad_fn(never_output.grad_fn, "CheckpointBackward") == 0


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def checkpoint_mode_invalid(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError, match="checkpoint is not one of 'always', 'except_last', or 'never'"):
        pipe_class(
            model,
            balance=[1],
            worker_map=get_worker_map(),
            chunks=2,
            checkpoint="INVALID_CHECKPOINT",
        )


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def checkpoint_mode_when_chunks_1(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))

    # All checkpoint modes are fine.
    pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=1,
        checkpoint="except_last",
    )
    pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=1, checkpoint="always")
    pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=1, checkpoint="never")


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def checkpoint_eval(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))
    model = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
    )
    input = torch.rand(2, 1)

    def find_grad_fn(grad_fn, name):
        if grad_fn is None:
            return False
        if grad_fn.__class__.__name__ == name:
            return True
        for next_grad_fn, _ in grad_fn.next_functions:
            if find_grad_fn(next_grad_fn, name):
                return True
        return False

    model.train()
    train_output = model(input)
    assert find_grad_fn(train_output.grad_fn, "CheckpointBackward")
    assert find_grad_fn(train_output.grad_fn, "RecomputeBackward")

    model.eval()
    eval_output = model(input)
    assert not find_grad_fn(eval_output.grad_fn, "CheckpointBackward")
    assert not find_grad_fn(eval_output.grad_fn, "RecomputeBackward")


@torch_spawn([2])
@pytest.mark.xfail(torch_version() < (1, 6, 0), reason="Doesn't work on torch < 1.6.0", strict=True)
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def checkpoint_non_float_input(pipe_class):
    class ForkNonFloat(nn.Module):
        def forward(self, input):
            return (input * 2, torch.tensor([False]))

    class JoinNonFloat(nn.Module):
        def forward(self, input):
            return input[0] * 2

    model = nn.Sequential(ForkNonFloat(), JoinNonFloat())
    model = pipe_class(
        model,
        balance=[1, 1],
        worker_map=get_worker_map(),
        chunks=1,
        checkpoint="always",
    )

    input = torch.rand(1, requires_grad=True)
    output = model(input)
    if model.group.rank() == 1:
        # with torch.autograd.detect_anomaly():
        output.backward()

    torch.distributed.barrier()


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def no_grad(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))
    model = pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=2)
    input = torch.rand(2, 1)

    latent = None

    def hook(module, input, output):
        _ = module
        _ = input

        nonlocal latent
        latent = output

    partition = model.partition
    partition.register_forward_hook(hook)

    with torch.no_grad():
        model(input)

    assert latent.grad_fn is None


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def exception(pipe_class):
    class ExpectedException(Exception):
        pass

    class Raise(nn.Module):
        def forward(self, *_):
            raise ExpectedException()

    model = nn.Sequential(Raise())
    model = pipe_class(model, balance=[1], worker_map=get_worker_map(), chunks=1)

    with pytest.raises(ExpectedException):
        model(torch.rand(1))


# FIXME(tom) should probably signal to all hosts in group to stop
@torch_spawn([4])
@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.device_count() < 4, reason="Not enough GPUs")
@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def exception_early_stop_asap(pipe_class):
    """Even the first partitions have finished to process, the partition before
    the failed partition hould be killed as soon as possible.
    """

    class ExpectedExceptio(Exception):
        pass

    class Pass(nn.Module):
        def forward(self, x):
            return x

    counter = 0

    class Counter(nn.Module):
        def forward(self, x):
            time.sleep(0.1)

            nonlocal counter
            counter += 1

            return x

    class Raise(nn.Module):
        def forward(self, x):
            raise ExpectedException()

    model = nn.Sequential(Pass(), Pass(), Counter(), Raise())
    model = pipe_class(model, [1, 1, 1, 1], worker_map=get_worker_map(), chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))

    # If the early stop doesn't work, it would be 3 instead.
    assert counter == 2


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def input_pair(pipe_class):
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_a = nn.Linear(1, 1)
            self.fc_b = nn.Linear(1, 1)

        def forward(self, a_and_b):
            a, b = a_and_b
            return (self.fc_a(a), self.fc_b(b))

    model = nn.Sequential(Two())
    model = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
    )

    a = torch.rand(10, 1, requires_grad=True)
    b = torch.rand(10, 1, requires_grad=True)

    a_out, b_out = model((a, b))
    loss = (a_out + b_out).mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def input_singleton(pipe_class):
    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, only_a):
            (a,) = only_a
            return (self.fc(a),)

    model = nn.Sequential(One())
    model = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
    )

    a = torch.rand(10, 1, requires_grad=True)

    (a_out,) = model((a,))
    loss = a_out.mean()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert a.grad is not None


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def input_varargs(pipe_class):
    model = nn.Sequential(nn.Linear(1, 1))
    model = pipe_class(model, balance=[1], worker_map=get_worker_map())

    a = torch.rand(1)
    b = torch.rand(1)

    # TypeError: forward() takes 2 positional arguments but 3 were given
    with pytest.raises(TypeError):
        model(a, b)


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def non_tensor(pipe_class):
    class NonTensor(nn.Module):
        def forward(self, _):
            return "hello"

    model = nn.Sequential(NonTensor())
    model = pipe_class(model, balance=[1], worker_map=get_worker_map())
    x = torch.rand(1)

    # TypeError: expected Tensor as element 0 in argument 0, but got str
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model("hello")


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def non_tensor_tuple(pipe_class):
    class NonTensorTuple(nn.Module):
        def forward(self, x):
            return (x, "hello")

    model = nn.Sequential(NonTensorTuple())
    model = pipe_class(model, balance=[1], worker_map=get_worker_map())
    x = torch.rand(1)

    # TypeError: CheckpointBackward.forward: expected Variable (got str) for return value 1
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model((x, "hello"))


@torch_spawn([1])
@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def deferred_batch_norm(checkpoint, lazy, pipe_class):
    bn = nn.BatchNorm2d(3)
    pipe_bn = deepcopy(bn)
    pipe_fn = lambda: pipe_bn  # noqa: E731
    if lazy:
        model = [LazyModule(pipe_fn)]
    else:
        model = nn.Sequential(pipe_bn)
    pipe = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint=checkpoint,
        deferred_batch_norm=True,
    )

    x = torch.rand(4, 3, 10, 10)
    pipe(x).mean().backward()
    bn(x).mean().backward()

    assert torch.allclose(pipe[0].running_mean, bn.running_mean, atol=1e-4)
    assert torch.allclose(pipe[0].running_var, bn.running_var, atol=1e-4)


@torch_spawn([1])
@pytest.mark.parametrize("checkpoint", ["never", "always"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def deferred_batch_norm_params(checkpoint, lazy, pipe_class):
    bn = nn.BatchNorm2d(3)
    pipe_bn = deepcopy(bn)
    pipe_fn = lambda: pipe_bn  # noqa: E731
    if lazy:
        model = [LazyModule(pipe_fn)]
    else:
        model = nn.Sequential(pipe_bn)
    pipe = pipe_class(
        model,
        balance=[1],
        worker_map=get_worker_map(),
        chunks=1,
        checkpoint=checkpoint,
        deferred_batch_norm=True,
    )

    x = torch.rand(4, 3, 10, 10)
    pipe(x).mean().backward()
    bn(x).mean().backward()

    assert pipe[0].weight.grad is not None
    assert pipe[0].bias.grad is not None

    assert torch.allclose(pipe[0].weight.grad, bn.weight.grad, atol=1e-4)
    assert torch.allclose(pipe[0].bias.grad, bn.bias.grad, atol=1e-4)


@torch_spawn([4])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def devices(pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)
    c = nn.Linear(1, 1)

    # There are extra two ranks.
    model = nn.Sequential(a, b, c)
    model = pipe_class(model, [1, 1, 1], worker_map=get_worker_map())

    # Extra devices must be discarded.
    if model.group.rank() == 3:
        assert model.pipeline is None


@torch_spawn([2])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def partitions(pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = pipe_class(model, [1, 1], worker_map=get_worker_map())

    assert isinstance(model.partition, nn.Sequential)

    if model.group.rank() == 0:
        assert model[0].weight == a.weight
    else:
        assert model[0].weight == b.weight


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def deny_moving(pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = pipe_class(model, [1, 1], worker_map=get_worker_map())

    model.cuda()
    model.cpu()
    model.to(torch.device("cuda"))
    model.to(0)
    model.to("cuda")
    model.to(device=0)
    model.to(torch.rand(1))
    model.to(tensor=torch.rand(1))

    # Casting is allowed.
    model.half()
    model.to(torch.double)
    model.to(dtype=torch.float)


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def empty_module(pipe_class):
    # Empty sequential module is not illegal.
    model = nn.Sequential()
    model = pipe_class(model, [], worker_map=get_worker_map())

    assert model(torch.tensor([42])) == torch.tensor([42])
    assert model((torch.tensor([42]),)) == (torch.tensor([42]),)

    # But only tensor or tensors is legal in MultiProcessPipe.

    with pytest.raises(TypeError):
        model(42)


@torch_spawn([2])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
@pytest.mark.skip(reason="TODO(msb) handle named_children")
def named_children(pipe_class):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(OrderedDict([("a", a), ("b", b)]))
    model = pipe_class(model, [1, 1], worker_map=get_worker_map())

    names = set(n for n, _ in model.named_modules())
    if model.group.rank() == 0:
        assert "0.a" in names
    else:
        assert "0.b" in names

    # MultiProcessPipe doesn't support __getattr__. Unlike nn.Sequential, MultiProcessPipe requires
    # several methods in its namespace.
    with pytest.raises(AttributeError):
        model.a


@torch_spawn([1])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def recommend_auto_balance(pipe_class):
    with pytest.raises(ValueError):
        # module and sum of balance have differen length (module: 0, sum of balance: 1)
        pipe_class(nn.Sequential(), [1])

    with pytest.raises(ValueError):
        # module and sum of balance have different length (module: 2, sum of balance: 1)
        pipe_class(nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)), [1])


@torch_spawn([2])
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def lazy_construction(pipe_class):
    init_count = 0

    class Custom(nn.Module):
        def __init__(self):
            super(Custom, self).__init__()
            nonlocal init_count
            init_count += 1

        def forward(self, x):
            return x

    model = [
        LazyModule(lambda: Custom()),
        LazyModule(lambda: Custom()),
        LazyModule(lambda: Custom()),
        LazyModule(lambda: Custom()),
    ]

    pipe = pipe_class(model, balance=[2, 2], worker_map=get_worker_map())

    assert isinstance(pipe[0], Custom)
    assert isinstance(pipe[1], Custom)
    assert len(pipe) == 2
    assert init_count == 2


@torch_spawn([2])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" in os.environ, reason="doesn't apply to mpi")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def missing_worker_map(pipe_class):
    model = nn.Sequential(nn.ReLU(), nn.ReLU())

    with pytest.raises(ValueError, match="'RpcTransport' requires 'worker_map' to be set"):
        pipe_class(model, [1, 1])


@torch_spawn([2])
@pytest.mark.skip(reason="currently broken")
@pytest.mark.parametrize("pipe_class", [AsyncPipe])
def verify_module_duplicate_parameters_on_distinct_partitions(pipe_class):
    class Surrogate(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(Surrogate(conv), Surrogate(conv))

    # FIXME(tom) can't have duplicate params with separate processes
    with pytest.raises(ValueError, match="module with duplicate parameters on distinct devices is not supported"):
        pipe_class(model, [1, 1], worker_map=get_worker_map())


@torch_spawn([4])
def async_event_loop():

    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU())
    pipe = AsyncPipe(model, [1, 1, 1, 1], worker_map=get_worker_map(), chunks=10)

    inputs = torch.rand(100, 10)

    output = pipe(inputs)
    if pipe.final_stage:
        loss = output.mean()
        loss.backward()
