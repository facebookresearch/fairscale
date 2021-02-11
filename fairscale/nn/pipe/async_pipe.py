# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor, nn

from fairscale.nn.model_parallel import get_pipeline_parallel_group

from . import microbatch
from .async_pipeline import AsyncPipeline
from .async_schedule import Invocation, Location, ModuleWrapper
from .batchnorm import DeferredBatchNorm
from .skip.layout import SkipLayout
from .skip.skippable import Skippable
from .types import LazyModule

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


@dataclass
class PartitionInfo:
    location: Location
    modules: "OrderedDict[str, nn.Module]"
    invocations: List[Invocation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.modules)


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


MOVING_DENIED = TypeError("denied to move parameters and buffers, because Pipe should manage device placement")


class AsyncPipe(Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train on Pipe_. If the module requires lots of memory, Pipe will be
    very efficient.

    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`AsyncPipe` module, as
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
    ) -> None:
        super().__init__()

        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        self.balance = list(balance)
        verify_module(module)
        check_balance(module, self.balance)

        self.chunks = chunks
        self.checkpoint = checkpoint
        self.pipeline: Optional[AsyncPipeline]
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

        self._skip_layout = SkipLayout(len(module), {})  # FIXME(tom)

        rank = self.group.rank()
        self.final_stage = rank == len(self.balance) - 1
        if rank >= len(self.balance):
            warnings.warn("More ranks than partitions, some ranks unused")
            self.partitions: List[ModuleWrapper] = []
            self.pipeline = None
            # TODO(msb) remove this hack
            self.partition = None
        else:
            self.partitions = self.instantiate_partition(module, self.balance, self.group)
            if deferred_batch_norm:
                for part in self.partitions:
                    part.module = DeferredBatchNorm.convert_deferred_batch_norm(part.module, chunks)
            for name, part in enumerate(self.partitions):
                self.add_module(str(name), part.module)
            self.create_pipeline()
            # TODO(msb) remove this hack
            self.partition = self.partitions[0].module

        del module

    def create_pipeline(self) -> None:
        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[self.checkpoint]

        self.pipeline = AsyncPipeline(
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
        """:class:`AsyncPipe` is a fairly transparent module wrapper. It doesn't
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
