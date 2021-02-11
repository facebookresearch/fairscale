# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict
from copy import deepcopy
import time

import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync import Pipe

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def test_parameters():
    model = nn.Sequential(nn.Linear(1, 1))
    pipe = Pipe(model, chunks=1)
    assert list(pipe.parameters()) != []


def test_public_attrs():
    class MyString:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return self.value

    model = nn.Sequential(nn.Linear(1, 1))
    pipe = Pipe(model, chunks=42.000, checkpoint=MyString("always"))

    assert pipe.devices == [torch.device("cpu")]
    assert pipe.chunks == 42
    assert isinstance(pipe.chunks, int)
    assert pipe.checkpoint == "always"
    assert isinstance(pipe.checkpoint, str)


def test_sequential_like():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = Pipe(model)

    assert len(model) == 2
    assert list(model) == [a, b]

    assert model[0] is a
    assert model[1] is b
    with pytest.raises(IndexError):
        _ = model[2]

    assert model[-1] is b
    assert model[-2] is a

def test_chunks_less_than_1():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError):
        Pipe(model, chunks=0)

    with pytest.raises(ValueError):
        Pipe(model, chunks=-1)

def test_batch_size_indivisible(setup_rpc):
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(7, 1))

    # Indivisible batch size is legal.
    assert not record


def test_batch_size_small(setup_rpc):
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(2, 1))

    # Batch size smaller than chunks is legal.
    assert not record


def test_checkpoint_mode(setup_rpc):
    def count_grad_fn(grad_fn, name, visited=None):
        if visited is None:
            visited = set()
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

    always = Pipe(model, chunks=2, checkpoint="always")
    except_last = Pipe(model, chunks=2, checkpoint="except_last")
    never = Pipe(model, chunks=2, checkpoint="never")

    always_output = always(input)
    except_last_output = except_last(input)
    never_output = never(input)

    assert count_grad_fn(always_output.local_value().grad_fn, "CheckpointBackward") == 2
    assert count_grad_fn(except_last_output.local_value().grad_fn, "CheckpointBackward") == 1
    assert count_grad_fn(never_output.local_value().grad_fn, "CheckpointBackward") == 0


def test_checkpoint_mode_invalid():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError, match="checkpoint is not one of 'always', 'except_last', or 'never'"):
        Pipe(model, chunks=2, checkpoint="INVALID_CHECKPOINT")


def test_checkpoint_mode_when_chunks_1():
    model = nn.Sequential(nn.Linear(1, 1))

    # All checkpoint modes are fine.
    Pipe(model, chunks=1, checkpoint="except_last")
    Pipe(model, chunks=1, checkpoint="always")
    Pipe(model, chunks=1, checkpoint="never")


def test_checkpoint_eval(setup_rpc):
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, chunks=2)
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
    assert find_grad_fn(train_output.local_value().grad_fn, "CheckpointBackward")
    assert find_grad_fn(train_output.local_value().grad_fn, "RecomputeBackward")

    model.eval()
    eval_output = model(input)
    assert not find_grad_fn(eval_output.local_value().grad_fn, "CheckpointBackward")
    assert not find_grad_fn(eval_output.local_value().grad_fn, "RecomputeBackward")


def test_checkpoint_non_float_input(setup_rpc):
    class ForkNonFloat(nn.Module):
        def forward(self, input):
            return (input * 2, torch.tensor([False]))

    class JoinNonFloat(nn.Module):
        def forward(self, input):
            return input[0] * 2

    model = nn.Sequential(ForkNonFloat(), JoinNonFloat())
    model = Pipe(model, chunks=1, checkpoint="always")

    input = torch.rand(1, requires_grad=True)
    output = model(input)
    output.backward()


def test_no_grad(setup_rpc):
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model, chunks=2)
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


def test_exception(setup_rpc):
    class ExpectedException(Exception):
        pass

    class Raise(nn.Module):
        def forward(self, *_):
            raise ExpectedException()

    model = nn.Sequential(Raise())
    model = Pipe(model, chunks=1)

    with pytest.raises(ExpectedException):
        model(torch.rand(1))


def test_exception_early_stop_asap(setup_rpc):
    """Even the first partitions have finished to process, the partition before
    the failed partition should be killed as soon as possible.
    """

    class ExpectedException(Exception):
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
    model = Pipe(model, chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))

    # If the early stop doesn't work, it would be 3 instead.
    assert counter == 2


def test_nested_input(setup_rpc):
    class NestedInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_a = nn.Linear(1, 1)
            self.fc_b = nn.Linear(1, 1)

        def forward(self, inp):
            return inp

    model = nn.Sequential(NestedInput())
    model = Pipe(model, chunks=2)

    a = torch.rand(10, 1, requires_grad=True)
    b = torch.rand(10, 1, requires_grad=True)

    # TypeError: expected Tensor, but got tuple
    with pytest.raises(TypeError):
        model((a, (a, b))).local_value()

    # TypeError: expected Tensor, but got list
    with pytest.raises(TypeError):
        model((a, [a, b])).local_value()


def test_input_pair(setup_rpc):
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_a = nn.Linear(1, 1)
            self.fc_b = nn.Linear(1, 1)

        def forward(self, a_and_b):
            a, b = a_and_b
            return (self.fc_a(a), self.fc_b(b))

    model = nn.Sequential(Two())
    model = Pipe(model, chunks=2)

    a = torch.rand(10, 1, requires_grad=True)
    b = torch.rand(10, 1, requires_grad=True)

    a_out, b_out = model((a, b)).local_value()
    loss = (a_out + b_out).mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None

    # Test with list.
    a.grad = None
    b.grad = None
    a_out, b_out = model([a, b]).local_value()
    loss = (a_out + b_out).mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None



def test_input_singleton(setup_rpc):
    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, only_a):
            (a,) = only_a
            return (self.fc(a),)

    model = nn.Sequential(One())
    model = Pipe(model, chunks=2)

    a = torch.rand(10, 1, requires_grad=True)

    (a_out,) = model((a,)).local_value()
    loss = a_out.mean()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert a.grad is not None

    # Test with list
    a.grad = None
    for p in model.parameters():
        p.grad = None

    (a_out,) = model([a]).local_value()
    loss = a_out.mean()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert a.grad is not None


def test_input_varargs(setup_rpc):
    model = nn.Sequential(nn.Linear(1, 1))
    model = Pipe(model)

    a = torch.rand(1)
    b = torch.rand(1)

    # TypeError: forward() takes 2 positional arguments but 3 were given
    with pytest.raises(TypeError):
        model(a, b)


def test_non_tensor(setup_rpc):
    class NonTensor(nn.Module):
        def forward(self, _):
            return "hello"

    model = nn.Sequential(NonTensor())
    model = Pipe(model)
    x = torch.rand(1)

    # TypeError: expected Tensor as element 0 in argument 0, but got str
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model("hello")


def test_non_tensor_sequence(setup_rpc):
    class NonTensorTuple(nn.Module):
        def forward(self, x):
            return (x, "hello")

    model = nn.Sequential(NonTensorTuple())
    model = Pipe(model)
    x = torch.rand(1)

    # TypeError: CheckpointBackward.forward: expected Variable (got str) for return value 1
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model((x, "hello"))

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model([x, "hello"])


@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
def test_deferred_batch_norm(checkpoint, setup_rpc):
    bn = nn.BatchNorm2d(3)
    pipe_bn = deepcopy(bn)
    pipe = Pipe(
        nn.Sequential(pipe_bn), chunks=2, checkpoint=checkpoint, deferred_batch_norm=True
    )

    x = torch.rand(4, 3, 10, 10)
    pipe(x).local_value().mean().backward()
    bn(x).mean().backward()

    assert torch.allclose(pipe[0].running_mean, bn.running_mean, atol=1e-4)
    assert torch.allclose(pipe[0].running_var, bn.running_var, atol=1e-4)


@pytest.mark.parametrize("checkpoint", ["never", "always"])
def test_deferred_batch_norm_params(checkpoint, setup_rpc):
    bn = nn.BatchNorm2d(3)
    pipe_bn = deepcopy(bn)
    pipe = Pipe(
        nn.Sequential(pipe_bn), chunks=1, checkpoint=checkpoint, deferred_batch_norm=True
    )

    x = torch.rand(4, 3, 10, 10)
    pipe(x).local_value().mean().backward()
    bn(x).mean().backward()

    assert pipe[0].weight.grad is not None
    assert pipe[0].bias.grad is not None

    assert torch.allclose(pipe[0].weight.grad, bn.weight.grad, atol=1e-4)
    assert torch.allclose(pipe[0].bias.grad, bn.bias.grad, atol=1e-4)


def test_devices():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)
    c = nn.Linear(1, 1)

    # There are extra two devices.
    model = nn.Sequential(a, b, c)
    model = Pipe(model)

    cpu = torch.device("cpu")
    # Extra devices must be discarded.
    assert model.devices == [cpu, cpu, cpu]


def test_partitions():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = Pipe(model)

    assert isinstance(model.partitions, nn.ModuleList)
    assert isinstance(model.partitions[0], nn.Sequential)
    assert isinstance(model.partitions[1], nn.Sequential)

    assert "partitions.0.0.weight" in model.state_dict()


def test_deny_moving():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = Pipe(model)

    # Moving is denied.
    with pytest.raises(TypeError):
        model.cuda()

    with pytest.raises(TypeError):
        model.cpu()

    with pytest.raises(TypeError):
        model.to(torch.device("cuda"))

    with pytest.raises(TypeError):
        model.to(0)

    with pytest.raises(TypeError):
        model.to("cuda")

    with pytest.raises(TypeError):
        model.to(device=0)

    with pytest.raises(TypeError):
        model.to(torch.rand(1))

    with pytest.raises(TypeError):
        model.to(tensor=torch.rand(1))

    # Casting is allowed.
    model.half()
    model.to(torch.double)
    model.to(dtype=torch.float)


def test_empty_module(setup_rpc):
    # Empty sequential module is not illegal.
    model = nn.Sequential()
    model = Pipe(model)

    assert model(torch.tensor(42)).local_value() == torch.tensor(42)
    assert model((torch.tensor(42),)).local_value() == (torch.tensor(42),)

    # But only tensor or tensors is legal in Pipe.
    with pytest.raises(TypeError):
        model(42)


def test_named_children():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(OrderedDict([("a", a), ("b", b)]))
    model = Pipe(model)

    names = set(n for n, _ in model.named_modules())
    assert "partitions.0.a" in names
    assert "partitions.1.b" in names

    # Pipe doesn't support __getattr__. Unlike nn.Sequential, Pipe requires
    # several methods in its namespace.
    with pytest.raises(AttributeError):
        model.a


def test_verify_module_non_sequential():
    with pytest.raises(TypeError, match="module must be nn.Sequential to be partitioned"):
        Pipe(nn.Module())


def test_verify_module_duplicate_children():
    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(conv, conv)

    with pytest.raises(ValueError, match="module with duplicate children is not supported"):
        Pipe(model)


@skip_if_no_cuda
def test_verify_module_params_on_same_device():
    class Surrogate(nn.Module):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

    conv1 = nn.Conv2d(3, 3, 1)
    conv2 = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(Surrogate(conv1, conv2.cuda()))

    with pytest.raises(
        ValueError,
        match=r'should have all parameters on a single device, please use .to\(\)'
            ' to place the module on a single device'):
        Pipe(model)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need atleast two GPUs")
def test_verify_nested_modules(setup_rpc):
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(32, 16).cuda(0),
            nn.Linear(16, 8).cuda(0)
        ),
        nn.Sequential(
            nn.Linear(8, 4).cuda(1),
            nn.Linear(4, 2).cuda(1)
        ),
    )

    pipe = Pipe(model)
    out = pipe(torch.rand(10, 32).cuda(0))
    assert out.local_value().device == torch.device("cuda:1")
    assert out.local_value().size() == torch.Size([10, 2])

def test_verify_module_duplicate_parameters_on_same_device():
    class Surrogate(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(Surrogate(conv), Surrogate(conv))

    Pipe(model)


def test_forward_lockstep(setup_rpc):
    timeline = []

    class DelayedLog(nn.Module):
        def __init__(self, j, seconds):
            super().__init__()
            self.i = 0
            self.j = j
            self.seconds = seconds

        def forward(self, x):
            time.sleep(self.seconds)

            timeline.append((self.i, self.j))
            self.i += 1

            return x

    model = nn.Sequential(DelayedLog(0, seconds=0), DelayedLog(1, seconds=0.1))
    model = Pipe(model, chunks=3)
    model(torch.rand(3, 1))

    # Expected timeline: (Logs are recorded at !)
    #
    # Partition #0: 0! 1!   2!
    # Partition #1:    000! 111! 222!
    #
    assert timeline == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1)]
