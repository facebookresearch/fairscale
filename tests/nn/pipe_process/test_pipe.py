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

from packaging import version
import pytest
import torch
from torch import nn

from fairscale.nn.model_parallel.initialize import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.pipe import Pipe
from tests.nn.model_parallel.commons import get_worker_map, torch_spawn


@torch_spawn([2])
def parameters():
    model = nn.Sequential(nn.Linear(1, 1))
    pipe = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=1)
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
        torch.distributed.send(t, 1)
    else:
        t = torch.empty(100).cuda()
        torch.distributed.recv(t, 0)

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
    group = torch.distributed.new_group([0, 1])
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
def public_attrs():
    class MyString:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return self.value

    model = nn.Sequential(nn.Linear(1, 1))

    pipe = Pipe(
        model,
        balance=(1,),
        style=Pipe.MultiProcess,
        worker_map=get_worker_map(),
        chunks=42.000,
        checkpoint=MyString("always"),
    )

    print(f"balance = {pipe.devices}")
    assert pipe.balance == [1]
    assert pipe.devices is None
    assert pipe.chunks == 42
    assert isinstance(pipe.chunks, int)
    assert pipe.checkpoint == "always"
    assert isinstance(pipe.checkpoint, str)


@torch_spawn([2])
@pytest.mark.parametrize("balance", [[2], [1, 1]])
def sequential_like(balance):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = Pipe(model, balance, style=Pipe.MultiProcess, worker_map=get_worker_map())

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
def balance_wrong_length():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map())

    with pytest.raises(ValueError):
        Pipe(model, balance=[3], style=Pipe.MultiProcess, worker_map=get_worker_map())


@torch_spawn([2])
def balance_less_than_1():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        Pipe(model, balance=[0, 2], style=Pipe.MultiProcess, worker_map=get_worker_map())

    with pytest.raises(ValueError):
        Pipe(model, balance=[-1, 3], style=Pipe.MultiProcess, worker_map=get_worker_map())


@torch_spawn([1])
def chunks_less_than_1():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError):
        Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=0)

    with pytest.raises(ValueError):
        Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=-1)


@torch_spawn([1])
def too_few_devices():
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))

    with pytest.raises(IndexError):
        # len(balance) > len(group.size())
        model = Pipe(model, balance=[1, 1, 1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())


@torch_spawn([1])
def batch_size_indivisible():
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(7, 1))

    # Indivisible batch size is legal.
    assert not record


@torch_spawn([1])
def batch_size_small():
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(2, 1))

    # Batch size smaller than chunks is legal.
    assert not record


@torch_spawn([1])
def checkpoint_mode():
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

    always = Pipe(
        model,
        balance=[1],
        style=Pipe.MultiProcess,
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint="always",
        pipelined_backward=False,
    )
    except_last = Pipe(
        model,
        balance=[1],
        style=Pipe.MultiProcess,
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint="except_last",
        pipelined_backward=False,
    )
    never = Pipe(
        model,
        balance=[1],
        style=Pipe.MultiProcess,
        worker_map=get_worker_map(),
        chunks=2,
        checkpoint="never",
        pipelined_backward=False,
    )

    always_output = always(input)
    except_last_output = except_last(input)
    never_output = never(input)

    assert count_grad_fn(always_output.grad_fn, "CheckpointBackward") == 2
    assert count_grad_fn(except_last_output.grad_fn, "CheckpointBackward") == 1
    assert count_grad_fn(never_output.grad_fn, "CheckpointBackward") == 0


@torch_spawn([1])
def checkpoint_mode_invalid():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError, match="checkpoint is not one of 'always', 'except_last', or 'never'"):
        Pipe(
            model,
            balance=[1],
            style=Pipe.MultiProcess,
            worker_map=get_worker_map(),
            chunks=2,
            checkpoint="INVALID_CHECKPOINT",
        )


@torch_spawn([1])
def checkpoint_mode_when_chunks_1():
    model = nn.Sequential(nn.Linear(1, 1))

    # All checkpoint modes are fine.
    Pipe(
        model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=1, checkpoint="except_last",
    )
    Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=1, checkpoint="always")
    Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=1, checkpoint="never")


@torch_spawn([1])
def checkpoint_eval():
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(
        model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=2, pipelined_backward=False,
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
@pytest.mark.xfail(
    version.parse(torch.__version__) < version.parse("1.6.0"), reason="Doesn't work on torch < 1.6.0", strict=True
)
def checkpoint_non_float_input():
    class ForkNonFloat(nn.Module):
        def forward(self, input):
            return (input * 2, torch.tensor([False]))

    class JoinNonFloat(nn.Module):
        def forward(self, input):
            return input[0] * 2

    model = nn.Sequential(ForkNonFloat(), JoinNonFloat())
    model = Pipe(
        model,
        balance=[1, 1],
        style=Pipe.MultiProcess,
        worker_map=get_worker_map(),
        chunks=1,
        checkpoint="always",
        pipelined_backward=False,
    )

    input = torch.rand(1, requires_grad=True)
    output = model(input)
    if model.group.rank() == 1:
        # with torch.autograd.detect_anomaly():
        output.backward()
    else:
        model.back_helper(output)


@torch_spawn([1])
def no_grad():
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=2)
    input = torch.rand(2, 1)

    latent = None

    def hook(module, input, output):
        _ = module
        _ = input

        nonlocal latent
        latent = output

    partition = model.partitions[0]
    partition.register_forward_hook(hook)

    with torch.no_grad():
        model(input)

    assert latent.grad_fn is None


@torch_spawn([1])
def exception():
    class ExpectedException(Exception):
        pass

    class Raise(nn.Module):
        def forward(self, *_):
            raise ExpectedException()

    model = nn.Sequential(Raise())
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=1)

    with pytest.raises(ExpectedException):
        model(torch.rand(1))


# FIXME(tom) should probably signal to all hosts in group to stop
@torch_spawn([4])
@pytest.mark.xfail(strict=True)
def exception_early_stop_asap():
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
    model = Pipe(model, [1, 1, 1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))

    # If the early stop doesn't work, it would be 3 instead.
    assert counter == 2


@torch_spawn([1])
def input_pair():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_a = nn.Linear(1, 1)
            self.fc_b = nn.Linear(1, 1)

        def forward(self, a_and_b):
            a, b = a_and_b
            return (self.fc_a(a), self.fc_b(b))

    model = nn.Sequential(Two())
    model = Pipe(
        model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=2, pipelined_backward=False,
    )

    a = torch.rand(10, 1, requires_grad=True)
    b = torch.rand(10, 1, requires_grad=True)

    a_out, b_out = model((a, b))
    loss = (a_out + b_out).mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None


@torch_spawn([1])
def input_singleton():
    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, only_a):
            (a,) = only_a
            return (self.fc(a),)

    model = nn.Sequential(One())
    model = Pipe(
        model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map(), chunks=2, pipelined_backward=False,
    )

    a = torch.rand(10, 1, requires_grad=True)

    (a_out,) = model((a,))
    loss = a_out.mean()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert a.grad is not None


@torch_spawn([1])
def input_varargs():
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map())

    a = torch.rand(1)
    b = torch.rand(1)

    # TypeError: forward() takes 2 positional arguments but 3 were given
    with pytest.raises(TypeError):
        model(a, b)


@torch_spawn([1])
def non_tensor():
    class NonTensor(nn.Module):
        def forward(self, _):
            return "hello"

    model = nn.Sequential(NonTensor())
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map())
    x = torch.rand(1)

    # TypeError: expected Tensor as element 0 in argument 0, but got str
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model("hello")


@torch_spawn([1])
def non_tensor_tuple():
    class NonTensorTuple(nn.Module):
        def forward(self, x):
            return (x, "hello")

    model = nn.Sequential(NonTensorTuple())
    model = Pipe(model, balance=[1], style=Pipe.MultiProcess, worker_map=get_worker_map())
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
def deferred_batch_norm(checkpoint, lazy):
    bn = nn.BatchNorm2d(3)
    pipe_bn = deepcopy(bn)
    pipe_fn = lambda: pipe_bn  # noqa: E731
    if lazy:
        model = [pipe_fn]
    else:
        model = nn.Sequential(pipe_bn)
    pipe = Pipe(
        model,
        balance=[1],
        style=Pipe.MultiProcess,
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
def deferred_batch_norm_params(checkpoint, lazy):
    bn = nn.BatchNorm2d(3)
    pipe_bn = deepcopy(bn)
    pipe_fn = lambda: pipe_bn  # noqa: E731
    if lazy:
        model = [pipe_fn]
    else:
        model = nn.Sequential(pipe_bn)
    pipe = Pipe(
        model,
        balance=[1],
        style=Pipe.MultiProcess,
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
def devices():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)
    c = nn.Linear(1, 1)

    # There are extra two ranks.
    model = nn.Sequential(a, b, c)
    model = Pipe(model, [1, 1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())

    # Extra devices must be discarded.
    if model.group.rank() == 3:
        assert model.pipeline is None


@torch_spawn([2])
def partitions():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = Pipe(model, [1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())

    assert isinstance(model.partitions, nn.ModuleList)
    assert len(model) == 1
    assert isinstance(model.partitions[0], nn.Sequential)

    assert "partitions.0.0.weight" in model.state_dict()


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def deny_moving():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = Pipe(model, [1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())

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
def empty_module():
    # Empty sequential module is not illegal.
    model = nn.Sequential()
    model = Pipe(model, [], style=Pipe.MultiProcess, worker_map=get_worker_map())

    assert model(torch.tensor([42])) == torch.tensor([42])
    assert model((torch.tensor([42]),)) == (torch.tensor([42]),)

    # But only tensor or tensors is legal in Pipe.

    with pytest.raises(TypeError):
        model(42)


@torch_spawn([2])
def named_children():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(OrderedDict([("a", a), ("b", b)]))
    model = Pipe(model, [1, 1], devices=["cpu", "cpu"])

    names = set(n for n, _ in model.named_modules())
    assert "partitions.0.a" in names
    assert "partitions.1.b" in names

    # Pipe doesn't support __getattr__. Unlike nn.Sequential, Pipe requires
    # several methods in its namespace.
    with pytest.raises(AttributeError):
        model.a


@torch_spawn([1])
def recommend_auto_balance():
    with pytest.raises(ValueError, match="fairscale.nn.pipe.balance"):
        # balance is required
        Pipe(nn.Sequential())

    with pytest.raises(ValueError, match="fairscale.nn.pipe.balance"):
        # module and sum of balance have differen length (module: 0, sum of balance: 1)
        Pipe(nn.Sequential(), [1])

    with pytest.raises(ValueError, match="fairscale.nn.pipe.balance"):
        # module and sum of balance have different length (module: 2, sum of balance: 1)
        Pipe(nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)), [1])


@torch_spawn([1])
def verify_module_non_sequential():
    with pytest.raises(TypeError, match="module must be nn.Sequential to be partitioned"):
        Pipe(nn.Module(), [1])


@torch_spawn([1])
def verify_module_duplicate_children():
    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(conv, conv)

    with pytest.raises(ValueError, match="module with duplicate children is not supported"):
        Pipe(model, [1, 1])


@torch_spawn([2])
def lazy_construction():
    init_count = 0

    class Custom(nn.Module):
        def __init__(self):
            super(Custom, self).__init__()
            nonlocal init_count
            init_count += 1

        def forward(self, x):
            return x

    model = [
        lambda: Custom(),
        lambda: Custom(),
        lambda: Custom(),
        lambda: Custom(),
    ]

    pipe = Pipe(model, balance=[2, 2], style=Pipe.MultiProcess, worker_map=get_worker_map())

    assert isinstance(pipe[0], Custom)
    assert isinstance(pipe[1], Custom)
    assert len(pipe) == 2
    assert init_count == 2


@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" in os.environ, reason="doesn't apply to mpi")
@torch_spawn([2])
def missing_worker_map():
    model = nn.Sequential(nn.ReLU(), nn.ReLU())

    with pytest.raises(ValueError, match="'PipelineStyle.MultiProcess' requires 'worker_map' to be set"):
        Pipe(model, [1, 1], style=Pipe.MultiProcess)


@torch_spawn([2])
@pytest.mark.skip(reason="currently broken")
def verify_module_duplicate_parameters_on_distinct_partitions():
    class Surrogate(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(Surrogate(conv), Surrogate(conv))

    # FIXME(tom) can't have duplicate params with separate processes
    with pytest.raises(ValueError, match="module with duplicate parameters on distinct devices is not supported"):
        Pipe(model, [1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())


@torch_spawn([4])
def pipelined_backward():
    model = nn.Sequential(nn.ReLU(), nn.ReLU())

    destroy_model_parallel()
    initialize_model_parallel(1, 4)
    pipe = Pipe(model, [1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())

    assert pipe.pipelined_backward is False

    destroy_model_parallel()
    initialize_model_parallel(2, 2)
    pipe = Pipe(model, [1, 1], style=Pipe.MultiProcess, worker_map=get_worker_map())

    assert pipe.pipelined_backward is True
