# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .async_pipeline import AsyncPipeline
from .async_schedule import Invocation, Location, ModuleWrapper
from .multiprocess_pipe import MultiProcessPipe
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


class AsyncPipe(MultiProcessPipe):
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
