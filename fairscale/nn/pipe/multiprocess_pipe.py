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

"""The MultiProcessPipe interface."""
from collections import OrderedDict
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda

from fairscale.nn.model_parallel import get_model_parallel_world_size, get_pipeline_parallel_group

from . import microbatch
from .async_schedule import Location, ModuleWrapper
from .batchnorm import DeferredBatchNorm
from .multiprocess_pipeline import MultiProcessPipeline
from .phony import get_phony
from .skip.layout import SkipLayout, inspect_skip_layout
from .skip.skippable import Skippable, verify_skippables
from .types import LazyModule

__all__ = ["MultiProcessPipe", "LazyModule"]


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict


def verify_module(module: Union[nn.Sequential, List[LazyModule]]) -> None:
    if len(set(map(id, module))) != len(module):
        raise ValueError("module with duplicate children is not supported")


def check_balance(module: Union[nn.Sequential, List[LazyModule]], balance: List[int]) -> None:
    if len(module) != sum(balance):
        raise ValueError(
            f"module and sum of balance have different length (module: {len(module)}, sum of balance: {sum(balance)})"
        )

    if any(x <= 0 for x in balance):
        raise ValueError(f"all balance numbers must be positive integer (balance: {balance})")


def split_module(module: nn.Sequential, balance: List[int]) -> List[nn.Sequential]:
    """Splits a module into multiple partitions.

    Returns:
        partitions

        Partitions are represented as a :class:`~torch.nn.ModuleList` whose
        item is a partition. All layers in a partition are placed in the
        same device.

    Raises:
        BalanceError:
            wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    j = 0
    partitions = []
    layers: NamedModules = OrderedDict()

    for name, layer in module.named_children():
        layers[name] = layer

        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            partition = nn.Sequential(layers)

            partitions.append(partition)

            # Prepare for the next partition.
            layers.clear()
            j += 1

    return partitions


MOVING_DENIED = TypeError("denied to move parameters and buffers, because Pipe should manage device placement")


class MultiProcessPipe(Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train on Pipe_. If the module requires lots of memory, Pipe will be
    very efficient.
    ::

        model = nn.Sequential(a, b, c, d)
        model = Pipe(model, balance=[1, 1, 1, 1], chunks=8)
        output = model(input)

    .. _Pipe: https://arxiv.org/abs/1811.06965

    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`Pipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    Args:
        module (torch.nn.Sequential):
            sequential module to be parallelized
        balance (ints):
            list of number of layers in each partition

    Keyword Args:
        group (ProcessGroup):
            the process group that all
            pipeline stages are a member of. Defaults to
            `get_pipeline_parallel_group()`
        worker_map (Dict[int, str]):
            a map from worker name (the first argument to
            `torch.distributed.rpc.init_rpc`) to global rank (i.e.
            `torch.distributed.get_rank()`) needed in order for pipeline stages
            to communicate with each other
        input_device (device):
            the device on which tensors should be located before being passed to
            the first module in a given pipeline stage
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)
        deferred_batch_norm (bool):
            whether to use deferred BatchNorm moving statistics (default:
            :data:`False`, see :class:`DeferredBatchNorm` for more
            details)
        pipelined_backward (bool, optional):
            if True, call torch.autograd.backward once per microbatch on the
            backward pass (instead of once for the whole batch). This works
            around a potential deadlock in pytorch when using tensor parallelism
            at the same time. Defaults to `True` if
            `get_model_parallel_world_size() > 1`
            (default: `None`)
        retain_graph (bool):
            The value passed to `torch.autograd.backwards(..., retain_graph=<value>)
            (default: = `True`)

    Raises:
        TypeError:
            the module is not a :class:`nn.Sequential <torch.nn.Sequential>`.
        ValueError:
            invalid arguments, or wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """

    #: The number of layers in each partition.
    balance: List[int] = []
    #                    ^^
    # The default value [] required for Sphinx's autoattribute.

    #: The devices mapped to each partition.
    #:
    #: ``devices[-1]`` refers to the device of the last partition, which means
    #: it is the output device. Probably, you need to use it to transfer the
    #: target to calculate the loss without a device mismatch
    #: :exc:`RuntimeError`. For example::
    #:
    #:     out_device = pipe.devices[-1]
    #:
    #:     for input, target in loader:
    #:         target = target.to(out_device, non_blocking=True)
    #:         output = pipe(input)
    #:         loss = F.cross_entropy(output, target)
    #:

    #: The number of micro-batches.
    chunks: int = 1

    #: The checkpoint mode to determine when to enable checkpointing. It is one
    #: of ``'always'``, ``'except_last'``, or ``'never'``.
    checkpoint: str = "except_last"

    def __init__(
        self,
        module: Union[nn.Sequential, List[LazyModule]],
        balance: Iterable[int],
        *,
        group: Optional[torch.distributed.ProcessGroup] = None,
        worker_map: Optional[Dict[int, str]] = None,
        input_device: Union[None, int, str, torch.device] = None,
        chunks: int = chunks,
        checkpoint: str = checkpoint,
        deferred_batch_norm: bool = False,
        pipelined_backward: bool = None,
        retain_graph: bool = False,
    ) -> None:
        super().__init__()

        chunks = int(chunks)
        checkpoint = str(checkpoint)

        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        self.balance = list(balance)
        verify_module(module)
        check_balance(module, self.balance)

        # Verify if the underlying skippable modules satisfy integrity. The
        # integrity can be verified before forward() because it is static.
        if isinstance(module, nn.Sequential):
            verify_skippables(module)

        self.chunks = chunks
        self.checkpoint = checkpoint
        self.pipelined_backward = pipelined_backward
        self.retain_graph = retain_graph
        self.pipeline: Optional[MultiProcessPipeline]
        self.lock = threading.Lock()

        self.worker_map = worker_map
        self.input_device = input_device

        self.group: torch.distributed.ProcessGroup
        if group is None:
            self.group = get_pipeline_parallel_group()
        else:
            self.group = group

        if self.group.size() < len(self.balance):
            raise IndexError(
                f"too few ranks to hold given partitions (ranks: {self.group.size()}, partitions:"
                f" {len(self.balance)})"
            )

        rank = self.group.rank()
        if rank >= len(self.balance):
            warnings.warn("More ranks than partitions, some ranks unused")
            self.partitions: List[ModuleWrapper] = []
        else:
            self.partitions = self.instantiate_partition(module, self.balance, self.group)
            if deferred_batch_norm:
                for part in self.partitions:
                    part.module = DeferredBatchNorm.convert_deferred_batch_norm(part.module, chunks)
            for name, part in enumerate(self.partitions):
                self.add_module(str(name), part.module)
        if isinstance(module, nn.Sequential):
            local_partitions = split_module(module, self.balance)
            self._skip_layout = inspect_skip_layout(local_partitions)
        else:
            self._skip_layout = SkipLayout(len(module), {})  # FIXME(tom)

        rank = self.group.rank()
        if rank >= len(self.balance):
            self.pipeline = None
            self.final_stage = False
        else:
            self.final_stage = rank == len(self.balance) - 1

            self.create_pipeline()
            del module
        if self.pipelined_backward is None:
            if get_model_parallel_world_size() > 1:
                self.pipelined_backward = True
            else:
                self.pipelined_backward = False

    def create_pipeline(self) -> None:
        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[self.checkpoint]

        self.pipeline = MultiProcessPipeline(
            self.partitions,
            self._skip_layout,
            checkpoint_stop,
            group=self.group,
            worker_map=self.worker_map,
            input_device=self.input_device,
            final_stage=self.final_stage,
        )

    def instantiate_partition(
        self, module: Union[nn.Sequential, List[LazyModule]], balance: List[int], group: torch.distributed.ProcessGroup,
    ) -> List[ModuleWrapper]:
        layers: NamedModules = OrderedDict()

        def maybe_realize(layer: Any) -> nn.Module:
            if isinstance(layer, nn.Module):
                return layer
            elif callable(layer):
                return layer()
            else:
                raise TypeError(f"layer must be nn.Module or callable, is {type(layer)}")

        def iterate_module(module: Union[nn.Sequential, list]) -> Iterable[Tuple[Any, nn.Module]]:
            if isinstance(module, nn.Sequential):
                yield from module.named_children()
            else:
                yield from ((str(k), v) for k, v in enumerate(module))

        j = 0

        for name, layer in iterate_module(module):
            layers[name] = layer

            if len(layers) == balance[j]:
                if j == group.rank():
                    for key in layers:
                        layers[key] = maybe_realize(layers[key])
                    if not isinstance(module, nn.Sequential):
                        for layer in layers.values():
                            if isinstance(layer, Skippable):
                                raise ValueError(
                                    "Can't use Skippable layers with multi-process pipe and lazy construction"
                                )

                    return [ModuleWrapper(nn.Sequential(layers), Location(j, 0))]

                # Prepare for the next partition.
                layers.clear()
                j += 1

        raise ValueError("Souldn't get here, more ranks than partitions")

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self.partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions: List[Any]
        partitions = self.partitions

        if index < 0:
            partitions = partitions[::-1]

        for partition in partitions:
            try:
                if isinstance(partition, ModuleWrapper):
                    return partition.module[index]
                else:
                    return partition[index]
            except IndexError:
                pass

            shift = len(partition)

            if index < 0:
                index += shift
            else:
                index -= shift

        raise IndexError

    def __iter__(self) -> Iterable[nn.Module]:
        """Iterates over children of the underlying sequential module."""
        for partition in self.partitions:
            yield from partition.module

    def forward(self, input: TensorOrTensors, *, event=None) -> TensorOrTensors:  # type: ignore
        """:class:`MultiProcessPipe` is a fairly transparent module wrapper. It doesn't
        modify the input and output signature of the underlying module. But
        there's type restriction. Input and output have to be a
        :class:`~torch.Tensor` or a tuple of tensors. This restriction is
        applied at partition boundaries too.

        Args:
            input (torch.Tensor or tensors): input mini-batch

        Returns:
            tensor or tensors: output mini-batch

        Raises:
            TypeError: input is not a tensor or tensors.

        """
        microbatch.check(input)

        if not self.pipeline:
            # No pipeline is not illegal, more ranks than partitions
            return input

        # Divide a mini-batch into micro-batches.
        batches = microbatch.scatter(input, self.chunks)

        # Run pipeline parallelism.
        with self.lock:
            self.pipeline.run(self.training, batches, event)

            if self.final_stage:
                # Merge the micro-batches into one mini-batch.
                if self.pipelined_backward:
                    with torch.no_grad():
                        output = microbatch.gather(batches)

                    phony = get_phony(
                        torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
                        requires_grad=True,
                    )
                    output = PipelinedBackwardPass.apply(output, batches, phony, True)  # self.retain_graph)
                else:
                    output = microbatch.gather(batches)
            else:
                # Don't merge micro-batches to avoid unnecessary edges in autograd
                # graph
                # FIXME(tom) should figure out a proper type here
                output = batches  # type: ignore

            return output

    def back_helper(self, output: List[microbatch.Batch]) -> None:
        if self.final_stage:
            raise ValueError("back_helper should only be called on non-final stages")

        if self.pipeline:
            self.pipeline.back_helper(output)


class PipelinedBackwardPass(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, input: TensorOrTensors, batches, phony, retain_graph) -> TensorOrTensors:
        ctx.batches = batches
        ctx.retain_graph = retain_graph
        return input

    @staticmethod
    # type: ignore
    def backward(ctx, *grads) -> Tuple:
        with torch.no_grad():
            grad_batches = microbatch.scatter(grads, len(ctx.batches))
        for grad, batch in reversed(list(zip(grad_batches, ctx.batches))):
            for t in batch:
                t.retain_grad()
            torch.autograd.backward(batch.tensor_or_tensors, grad_tensors=(*grad,), retain_graph=ctx.retain_graph)

        with torch.no_grad():
            if ctx.batches[0].atomic:
                tensors = tuple(b.tensor.grad for b in ctx.batches)
                output: TensorOrTensors = torch.cat(tensors)
            else:
                rotated = [[t.grad for t in b.tensors] for b in ctx.batches]
                output_buf = []

                for tensors in zip(*rotated):
                    output_buf.append(torch.cat(tensors))

                output = tuple(output_buf)
            del ctx.batches

        return (output, None, None, None)
