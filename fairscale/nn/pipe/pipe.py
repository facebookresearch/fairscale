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

"""The Pipe interface."""
from collections import OrderedDict
import itertools
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, cast
import warnings

from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda

from fairscale.nn.model_parallel import get_model_parallel_world_size, get_pipeline_parallel_group

from . import microbatch
from .async_schedule import Invocation, Location, ModuleWrapper
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import SkipLayout, inspect_skip_layout
from .skip.skippable import Skippable, verify_skippables
from .stream import AbstractStream, new_stream
from .types import LazyModule, PipelineStyle

__all__ = ["Pipe", "LazyModule"]


Device = Union[torch.device, int, str]
Devices = Union[Iterable[Device], List[Device]]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

ListOfLazyModules = List[LazyModule]

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict


def recommend_auto_balance(message: str) -> str:
    """Expands a message with recommendation to :mod:`torchpipe.balance`."""
    return f"""{message}

If your model is still under development, its optimal balance would change
frequently. In this case, we highly recommend 'fairscale.nn.pipe.balance' for
naive automatic balancing:

  from fairscale.nn import Pipe
  from fairscale.nn.pipe.balance import balance_by_time

  partitions = torch.cuda.device_count()
  sample = torch.empty(...)
  balance = balance_by_time(partitions, model, sample)

  model = Pipe(model, balance, ...)
"""


# FIXME(tom) make this a valid way to call
def verify_list_of_callable(module: Union[nn.Sequential, list]) -> None:
    for layer in module:
        if isinstance(layer, nn.Module):
            pass
        elif isinstance(layer, LazyModule):
            pass
        else:
            raise TypeError(f"layer {type(layer)} must be nn.Module or LazyModule to be partitioned")


def verify_module(module: Union[nn.Sequential, ListOfLazyModules]) -> None:
    if isinstance(module, Iterable) and not isinstance(module, nn.Sequential):
        verify_list_of_callable(module)
    else:
        if not isinstance(module, nn.Sequential):
            raise TypeError("module must be nn.Sequential to be partitioned")

        named_children = list(module.named_children())
        if len(named_children) != len(module):
            raise ValueError("module with duplicate children is not supported")


def verify_splitting(
    module: nn.Sequential,
    partitions: List[nn.Sequential],
    balance: Iterable[int],
    devices: Optional[List[torch.device]],
) -> None:
    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum(len(list(child.parameters())) for child in module.children())
    if num_parameters == num_child_parameters:
        return

    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            parti = partitions[i]
            partj = partitions[j]
            if devices and devices[i] == devices[j]:
                continue
            for p in parti.parameters():
                for q in partj.parameters():
                    if p is q:
                        raise ValueError("module with duplicate parameters on distinct devices is not supported")


class BalanceError(ValueError):
    pass


def check_balance(module: Any, balance: Iterable[int], filter_unique: bool = False) -> None:

    if filter_unique:
        module_len = len(set(map(id, module)))
    else:
        module_len = len(module)

    if module_len != sum(balance):
        raise BalanceError(
            f"module and sum of balance have different length (module: {len(module)}, sum of balance: {sum(balance)})"
        )

    if any(x <= 0 for x in balance):
        raise BalanceError(f"all balance numbers must be positive integer (balance: {balance})")


@dataclass
class PartitionInfo:
    location: Location
    modules: "OrderedDict[str, nn.Module]"
    invocations: List[Invocation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.modules)


def instantiate_partition(
    module: Union[nn.Sequential, ListOfLazyModules],
    balance: Iterable[int],
    group: torch.distributed.ProcessGroup,
    style: PipelineStyle,
) -> List[ModuleWrapper]:
    balance = list(balance)
    check_balance(module, balance, True)

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

    if style == PipelineStyle.AsyncSchedule:
        module_ids = list(map(id, module))
        index_of_first_use = [module_ids.index(x) for x in module_ids]
        locations: List[Location] = []
        module_iter = enumerate(iterate_module(module))

        partitions: List[List[PartitionInfo]] = []
        for bi, b in enumerate(balance):
            modules_for_rank: List[PartitionInfo] = []
            current_module: OrderedDict[str, nn.Module] = OrderedDict()

            def current_location() -> Location:
                return Location(bi, len(modules_for_rank))

            def append_module(mod: "OrderedDict[str, nn.Module]") -> None:
                modules_for_rank.append(PartitionInfo(current_location(), mod))

            while sum(map(len, modules_for_rank)) + len(current_module) < b:
                module_index, (name, layer) = next(module_iter)

                if index_of_first_use[module_index] != module_index:
                    # Subsequent reuse of a module
                    locations.append(locations[index_of_first_use[module_index]])
                    continue

                is_reused = index_of_first_use.count(index_of_first_use[module_index]) > 1

                if is_reused and len(current_module) > 0:
                    append_module(current_module)
                    current_module = OrderedDict()

                current_module[str(name)] = layer
                locations.append(current_location())

                if is_reused:
                    append_module(current_module)
                    current_module = OrderedDict()

            if len(current_module) > 0:
                append_module(current_module)

            partitions.append(modules_for_rank)

        filtered_locations: List[Optional[Location]] = [loc for loc, _ in itertools.groupby(locations)]
        filtered_locations.append(None)

        for i in range(len(filtered_locations) - 1):
            loc = filtered_locations[i]
            assert loc
            if i == 0:
                inv = Invocation(i, loc, None, filtered_locations[i + 1])
            else:
                inv = Invocation(i, loc, filtered_locations[i - 1], filtered_locations[i + 1])

            partitions[loc.stage][loc.index].invocations.append(inv)

        invocations = enumerate(iterate_module(module))

        partition = partitions[group.rank()]
        result: List[ModuleWrapper] = []
        for partition_info in partition:
            wrapper = ModuleWrapper(
                nn.Sequential(OrderedDict((k, maybe_realize(m)) for k, m in partition_info.modules.items())),
                partition_info.location,
                partition_info.invocations,
            )

            if not isinstance(module, nn.Sequential):
                for layer in wrapper.module:
                    if isinstance(layer, Skippable):
                        raise ValueError("Can't use Skippable layers with multi-process pipe and lazy construction")

            result.append(wrapper)

        return result

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
                            raise ValueError("Can't use Skippable layers with multi-process pipe and lazy construction")

                return [ModuleWrapper(nn.Sequential(layers), Location(j, 0))]

            # Prepare for the next partition.
            layers.clear()
            j += 1

    raise ValueError("Souldn't get here, more ranks than partitions")


def split_module(
    module: nn.Sequential, balance: Iterable[int], devices: Optional[List[torch.device]],
) -> Tuple[List[nn.Sequential], List[int], Optional[List[torch.device]]]:
    """Splits a module into multiple partitions.

    Returns:
        A tuple of (partitions, balance, devices).

        Partitions are represented as a :class:`~torch.nn.ModuleList` whose
        item is a partition. All layers in a partition are placed in the
        same device.

    Raises:
        BalanceError:
            wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance = list(balance)

    check_balance(module, balance)

    if devices and len(balance) > len(devices):
        raise IndexError(
            f"too few devices to hold given partitions (devices: {len(devices)}, partitions: {len(balance)})"
        )

    j = 0
    partitions = []
    layers: NamedModules = OrderedDict()

    for name, layer in module.named_children():
        layers[name] = layer

        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            partition = nn.Sequential(layers)

            if devices:
                device = devices[j]
                partition.to(device)

            partitions.append(partition)

            # Prepare for the next partition.
            layers.clear()
            j += 1

    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    if devices:
        del devices[j:]

    return partitions, balance, devices


MOVING_DENIED = TypeError("denied to move parameters and buffers, because Pipe should manage device placement")


class Pipe(Module):
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
        style (PipelineStyle):
            whether to use a single process for all pipeline stages or to assign
            one stage per process
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        group (ProcessGroup):
            specific to `style=MultiProcess`, the process group that all
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

    SingleProcess: PipelineStyle = PipelineStyle.SingleProcess
    MultiProcess: PipelineStyle = PipelineStyle.MultiProcess
    AsyncSchedule: PipelineStyle = PipelineStyle.AsyncSchedule

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
    devices: Optional[List[torch.device]] = None

    #: The number of micro-batches.
    chunks: int = 1

    #: The checkpoint mode to determine when to enable checkpointing. It is one
    #: of ``'always'``, ``'except_last'``, or ``'never'``.
    checkpoint: str = "except_last"

    def __init__(
        self,
        module: Union[nn.Sequential, ListOfLazyModules],
        balance: Optional[Iterable[int]] = None,
        *,
        style: PipelineStyle = PipelineStyle.SingleProcess,
        devices: Optional[Devices] = None,
        group: Optional[torch.distributed.ProcessGroup] = None,
        worker_map: Optional[Dict[int, str]] = None,
        input_device: Union[None, int, str, torch.device] = None,
        chunks: int = chunks,
        checkpoint: str = checkpoint,
        deferred_batch_norm: bool = False,
        pipelined_backward: bool = None,
        retain_graph: bool = False,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        chunks = int(chunks)
        checkpoint = str(checkpoint)

        if balance is None:
            raise ValueError(recommend_auto_balance("balance is required"))
        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        verify_module(module)

        # Verify if the underlying skippable modules satisfy integrity. The
        # integrity can be verified before forward() because it is static.
        if isinstance(module, nn.Sequential):
            verify_skippables(module)

        self.chunks = chunks
        self.checkpoint = checkpoint
        self.pipelined_backward = pipelined_backward
        self.retain_graph = retain_graph
        self.pipeline: Optional[Pipeline]
        self.loss_fn = loss_fn
        self.lock = threading.Lock()

        self.group = group
        self.worker_map = worker_map
        self.input_device = input_device

        self._copy_streams: List[List[AbstractStream]] = []

        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[self.checkpoint]

        if style is PipelineStyle.SingleProcess:
            module = cast(nn.Sequential, module)
            if deferred_batch_norm:
                module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)

            if input_device is not None:
                raise ValueError("'input_device' argument only applies to 'PipelineStyle.MultiProcess'")

            if devices is None:
                devices = range(torch.cuda.device_count())

            devices = [torch.device(d) for d in devices]
            devices = cast(List[torch.device], devices)

            try:
                self.partitions, self.balance, self.devices = split_module(module, balance, devices)
            except BalanceError as exc:
                raise ValueError(recommend_auto_balance(str(exc)))
            verify_splitting(module, self.partitions, self.balance, self.devices)

            self._skip_layout = inspect_skip_layout(self.partitions)

            # Separate CUDA streams for copy.
            copy_streams = self._ensure_copy_streams()
            if self.pipelined_backward is None:
                self.pipelined_backward = False
            self.pipeline = Pipeline(
                self.partitions, self.devices, copy_streams, self._skip_layout, checkpoint_stop, style=style,
            )

        elif style in [PipelineStyle.MultiProcess, PipelineStyle.AsyncSchedule]:

            if self.group is None:
                self.group = get_pipeline_parallel_group()
            assert self.group

            if devices is not None:
                raise ValueError("'devices' argument only applies to 'PipelineStyle.SingleProcess'")

            self.balance = list(balance)

            if self.group.size() < len(self.balance):
                raise IndexError(
                    f"too few ranks to hold given partitions (ranks: {self.group.size()}, partitions:"
                    f" {len(self.balance)})"
                )
            try:
                rank = self.group.rank()
                if rank >= len(self.balance):
                    warnings.warn("More ranks than partitions, some ranks unused")
                    self.mp_partitions: List[ModuleWrapper] = []
                else:
                    self.mp_partitions = instantiate_partition(module, balance, self.group, style)
                    if deferred_batch_norm:
                        for part in self.mp_partitions:
                            part.module = DeferredBatchNorm.convert_deferred_batch_norm(part.module, chunks)
                    for name, part in enumerate(self.mp_partitions):
                        self.add_module(str(name), part.module)
                self.devices = None
                if isinstance(module, nn.Sequential):
                    local_partitions, _, _ = split_module(module, balance, None)
                    self._skip_layout = inspect_skip_layout(local_partitions)
                else:
                    self._skip_layout = SkipLayout(len(module), {})  # FIXME(tom)

            except BalanceError as exc:
                raise ValueError(recommend_auto_balance(str(exc)))

            rank = self.group.rank()
            if rank >= len(self.balance):
                self.pipeline = None
                self.final_stage = False
            else:
                self.final_stage = rank == len(self.balance) - 1
                assert loss_fn is None or self.final_stage

                self.pipeline = Pipeline(
                    cast(List[nn.Sequential], self.mp_partitions),
                    None,
                    None,
                    self._skip_layout,
                    checkpoint_stop,
                    style=style,
                    group=self.group,
                    worker_map=self.worker_map,
                    input_device=self.input_device,
                    final_stage=self.final_stage,
                )
                del module
            if self.pipelined_backward is None:
                if get_model_parallel_world_size() > 1:
                    self.pipelined_backward = True
                else:
                    self.pipelined_backward = False

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        if hasattr(self, "partitions"):
            return sum(len(p) for p in self.partitions)
        else:
            return sum(len(p) for p in self.mp_partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions: List[Any]
        if hasattr(self, "partitions"):
            partitions = self.partitions
        else:
            partitions = self.mp_partitions

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
        if hasattr(self, "partitions"):
            for partition in self.partitions:
                yield from partition
        else:
            for mp_partition in self.mp_partitions:
                yield from mp_partition.module

    # Pipe should manage the device of each partition.
    # Deny cuda(), cpu(), and to() with device, by TypeError.
    def cuda(self, device: Optional[Device] = None) -> "Pipe":
        if self.devices:
            raise MOVING_DENIED
        if device:
            return super().cuda(device=device)
        else:
            return super().cuda()

    def cpu(self) -> "Pipe":
        if self.devices:
            raise MOVING_DENIED
        return super().cpu()

    def to(self, *args: Any, **kwargs: Any) -> "Pipe":
        """Restrict .to() options.

        Deny these usages:
         - to(device[, dtype, non_blocking])
         - to(tensor[, non_blocking])

         But allow this:
         - to(dtype[, non_blocking])
        """
        if self.devices:
            if "device" in kwargs or "tensor" in kwargs:
                raise MOVING_DENIED

            if args:
                if isinstance(args[0], (torch.device, int, str)):
                    raise MOVING_DENIED
                if torch.is_tensor(args[0]):
                    raise MOVING_DENIED

        return super().to(*args, **kwargs)

    def _ensure_copy_streams(self) -> List[List[AbstractStream]]:
        """Ensures that :class:`Pipe` caches CUDA streams for copy.

        It's worth to cache CUDA streams although PyTorch already manages a
        pool of pre-allocated CUDA streams, because it may reduce GPU memory
        fragementation when the number of micro-batches is small.

        """
        if not self._copy_streams:
            assert self.devices is not None
            for device in self.devices:
                self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])

        return self._copy_streams

    def forward(self, input: TensorOrTensors, *, event=None) -> TensorOrTensors:  # type: ignore
        """:class:`Pipe` is a fairly transparent module wrapper. It doesn't
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

        if not self.group and not self.devices:
            # Empty sequential module is not illegal.
            return input

        if not self.pipeline:
            # No pipeline is not illegal, more ranks than partitions
            return input

        # Divide a mini-batch into micro-batches.
        batches = microbatch.scatter(input, self.chunks)

        # Run pipeline parallelism.
        with self.lock:
            self.pipeline.run(self.training, batches, event)

            if self.group and not self.final_stage:
                # Don't merge micro-batches to avoid unnecessary edges in autograd
                # graph
                # FIXME(tom) should figure out a proper type here
                return batches  # type: ignore
            else:
                # Merge the micro-batches into one mini-batch.
                if self.pipelined_backward:
                    with torch.no_grad():
                        output = microbatch.gather(batches)

                    from .phony import get_phony

                    phony = get_phony(
                        torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
                        requires_grad=True,
                    )
                    output = PipelinedBackwardPass.apply(output, batches, phony, True)  # self.retain_graph)
                else:
                    output = microbatch.gather(batches)

            return output

    def back_helper(self, output: List[microbatch.Batch]) -> None:
        if self.final_stage:
            raise ValueError("back_helper should only be called on non-final stages")

        if self.pipeline:
            self.pipeline.back_helper(list(reversed(output)))


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
