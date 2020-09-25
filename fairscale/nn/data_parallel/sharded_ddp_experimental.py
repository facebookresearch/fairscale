# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A distributed data parallel class that shards the model into pieces
"""

from typing import Any, Dict, Iterable, List, Type, cast

import torch
from torch import Tensor, nn


def _slice_module(module: nn.Sequential, number_shards: int) -> List[List[nn.Module]]:
    # Naive sharding for now, slice by the number of layers
    # This is probably suboptimal if the complexity or size of the layers vary by a lot
    def chunks(lst: List[nn.Module], n: int) -> Iterable[List[nn.Module]]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    return list(chunks(list(module.modules()), number_shards))


class ShardedDataParallel(nn.Module):
    """Implements distributed data parallel training with optimizer state sharding.

    This experiments with a novel way to get to the full zero suite
    The model is sharded, and we create a process group per shard.

    Args:
        module (~torch.nn.Sequential): module to be parallelized
        optimizer (~torch.optim.Optimizer): optimizer to be used for training
        optimizer_params(Dict): extra parameters for the optimizer
        world_size (int): number of parallel workers
        process_group (optional): the c10d process group to be used for
            distributed gradient reduction. If None, the default WORLD process group
            will be used.
    """

    def __init__(
        self,
        module: nn.Sequential,  # hard pre-requisite for now, easier model slicing
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict[str, Any],
        world_size: int,
        process_group: Any = None,
    ):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group if process_group is not None else torch.distributed.group.WORLD
        self.rank = torch.distributed.get_rank(self.process_group)
        self.backend = torch.distributed.get_backend(group=self.process_group)  # type: ignore

        # Slice the model
        module_slices = _slice_module(module, self.world_size)

        # Create one data parallel process group per shard.
        self.ddp_shards: List[nn.Module] = []

        for i_slice, module_shard in enumerate(module_slices):
            # Authoritative rank moves with the slices
            shard_rank = (i_slice + self.rank) % self.world_size

            # Create one group per slice
            pg = torch.distributed.init_process_group(  # type: ignore
                backend=self.backend,
                init_method="env://",
                rank=shard_rank,
                world_size=self.world_size,
                group_name=f"shard_{i_slice}",
            )

            # Data parallel within this group
            ddp = torch.nn.parallel.DistributedDataParallel(nn.Sequential(*module_shard), process_group=pg)

            # Three specific hooks to register:
            # - All ranks other than the authoritative one can let go of the parameters after the FW
            # pass, DDP will re-broadcast them at the next step
            # TODO: Ben

            # - Bring the grads back to the authoritative rank (reduce)
            # TODO: Ben

            # - All ranks other than the authoritative one can let go of the grads after the FW
            # pass, DDP will re-create them for the next BW
            # TODO: Ben

            self.ddp_shards.append(ddp)

            # Use one normal optimizer per shard
            if i_slice == self.rank:
                self.optim = optimizer(nn.Sequential(*module_shard).parameters(), **optimizer_params)

    def forward(self, *inputs: Any, **kwargs: Any) -> Tensor:
        # Go through the data parallel instances
        # The batch is spread during the FW pass
        for ddp in self.ddp_shards:
            inputs = ddp(*inputs, **kwargs)

        return cast(Tensor, inputs)  # FIXME
