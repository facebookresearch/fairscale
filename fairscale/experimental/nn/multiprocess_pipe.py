# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import itertools
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Type, Union

import torch
from torch import Tensor
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

LayerSpec = Tuple[str, Type[nn.Module], Tuple, Dict]

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
else:
    Module = nn.Module

if torch.__version__.split("+")[0].split(".")[:3] <= ["1", "8", "1"]:
    BOUNCE_TENSORS = True
else:
    BOUNCE_TENSORS = False


def _verify_module(module: List[LayerSpec]) -> None:
    if not isinstance(module, List):
        raise TypeError("module must be a list")

    for elem in module:
        if not isinstance(elem, tuple):
            raise TypeError("module must be a list of tuple")
        if len(elem) != 4:
            raise TypeError("each module tuple must contain (name, nn.module, args, kwargs)")
        name, layer, args, kwargs = elem
        if not (
            isinstance(name, str)
            and issubclass(layer, nn.Module)
            and isinstance(args, tuple)
            and isinstance(kwargs, dict)
        ):
            raise TypeError("each module tuple must contain (name, nn.module, args, kwargs)")


class _ToHere(Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def forward(self, x_rref: rpc.RRef) -> Tensor:  # type: ignore
        if BOUNCE_TENSORS:
            return x_rref.remote().cpu().to_here().to(self.device)
        else:
            return x_rref.to_here()


def _create_sequential(layer_spec: List[LayerSpec], device: str) -> Module:
    layers = [(name, layer(*args, **kwargs)) for name, layer, args, kwargs in layer_spec]  # type: ignore
    layers.insert(0, ("to_here", _ToHere(device)))
    return nn.Sequential(OrderedDict(layers)).to(device)


def _rcat(tensors: List) -> Tensor:
    return torch.cat([t.local_value() for t in tensors])


def _rcheckpoint(rmodule: rpc.RRef, input_rref: rpc.RRef) -> TensorOrTensors:
    module = rmodule.local_value()
    input = module[0](input_rref)  # calls _ToHere.forward
    return checkpoint_sequential(module[1:], 1, input)


def _parameter_rrefs(module: rpc.RRef) -> List[rpc.RRef]:
    return [rpc.RRef(p) for p in module.local_value().parameters()]


def rloss(loss_func: Callable, input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
    if BOUNCE_TENSORS:
        return loss_func(input_rref.remote().cpu().to_here(), target_rref.remote().cpu().to_here())
    else:
        return loss_func(input_rref.to_here(), target_rref.to_here())


def DistributedLoss(loss: nn.Module, *args: Tuple, **kwargs: Dict) -> Callable:
    loss_func = loss(*args, **kwargs)

    def dloss(input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
        return rpc.remote(input_rref.owner(), rloss, args=(loss_func, input_rref, target_rref))

    return dloss


class MultiProcessPipe(Module):
    """Paritions a sequential list of modules across multiple workers to train using a
    MultiProcessPipe_. If the module requires lots of memory, MultiProcessPipe will be
    very efficient.
    ::

        devices = ["worker0/cuda:0", "worker1/cuda:1"]
        # LayerSpec format: [("name", nn.Module, args, kwargs), ...]
        layer_spec = [("linear", nn.Linear, (4, 4), {}), ("relu", nn.ReLU, (), {})]
        pipe = MultiProcessPipe(layer_spec, balance=[1, 1], chunks=4, devices=devices)
        output = model(input)

    .. _MultiProcessPipe: https://arxiv.org/abs/1811.06965

    MultiProcessPipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`MultiProcessPipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    Args:
        module (list of LayerSpec):
            LayerSpec is a tuple constructed as follows: (name, nn.Module, args, kwargs)

    Keyword Args:
        balance (ints):
            list of number of layers in each partition
        devices (list of devices):
            rpc devices to use (e.g. ``["worker0/cuda:0", "worker1/cuda:1"]``)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)

    Raises:
        TypeError:
            the module is not a proper LayerSpec.
        ValueError:
            invalid arguments, or wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """

    def __init__(
        self,
        module: List[LayerSpec],
        *,
        balance: List[int],
        devices: List[str],
        chunks: int = 1,
        checkpoint: str = "never",
    ) -> None:
        super().__init__()

        if type(chunks) is not int or chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")
        if len(balance) != len(devices):
            raise IndexError("balance and devices lists must be the same size")
        if len(module) != sum(balance):
            raise IndexError("number of layers must match aggregate balance")

        _verify_module(module)

        index = 0
        rmodule = []
        workers = []
        for num_layers, device_spec in zip(balance, devices):
            worker, device = device_spec.split("/")
            next_index = index + num_layers
            rlayer = rpc.remote(worker, _create_sequential, args=(module[index:next_index], device))
            index = next_index
            workers.append(worker)
            rmodule.append(rlayer)

        # The micro-batch index where the checkpointing stops.
        self.checkpoint_stop = {"always": chunks, "except_last": chunks - 1, "never": 0}[checkpoint]

        self.chunks = chunks
        self.checkpoint = checkpoint
        self.module = module
        self.workers = workers
        self.rmodule = rmodule

    def forward(self, x: Tensor) -> rpc.RRef:  # type: ignore
        outputs = []
        for i, chunk in enumerate(x.chunk(self.chunks)):
            output = rpc.RRef(chunk)
            if i < self.checkpoint_stop:
                for rlayer in self.rmodule:
                    output = rpc.remote(rlayer.owner(), _rcheckpoint, args=(rlayer, output))
            else:
                for rlayer in self.rmodule:
                    output = rlayer.remote().forward(output)
            outputs.append(output)
        return rpc.remote(outputs[0].owner(), _rcat, args=(outputs,))

    def parameter_rrefs(self) -> List[rpc.RRef]:
        rrefs_list_of_lists = [rpc.rpc_sync(l.owner(), _parameter_rrefs, args=(l,)) for l in self.rmodule]
        return list(itertools.chain(*rrefs_list_of_lists))
