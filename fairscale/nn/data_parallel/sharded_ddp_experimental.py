# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A distributed data parallel class that shards the model and the optimizer into pieces.

See https://github.com/pytorch/pytorch/issues/42849 for more context. Credits to Shen Li for the original idea

"""

from typing import Any, Dict, List, Type

import torch
from torch import nn
import torch.distributed as dist


def _split(modules: nn.Sequential, number_shards: int) -> List[List[nn.Module]]:
    # Naive sharding for now, slice by the number of layers
    # This is probably suboptimal if the complexity or size of the layers vary by a lot
    splits: List[List[nn.Module]] = [[] for _ in range(number_shards)]
    i = 0
    n = len(modules) // number_shards

    print(f"Aiming for {n} blocks per shard")

    for m in modules:
        if splits and len(splits[i]) == n and (i < number_shards - 1):
            i += 1

        splits[i].append(m)

    return splits


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the fly for the FW pass and gather gradients.
    Depending on whether this rank is or is not the `owner_rank`, this ModelShard either only handles
    a shard of the compute and is stateless or also owns the up to date state.
    """

    def __init__(self, cpu_model_shard: nn.Module, owner_rank: int, process_group: Any):
        super().__init__()
        self.owner_rank = owner_rank
        self.process_group = process_group
        self.model_shard = cpu_model_shard
        self.is_owner = dist.get_rank(self.process_group) == self.owner_rank
        self.world_size = dist.get_world_size(self.process_group)

        # Save all the parameter sizes to be able to restore them
        if not self.is_owner:
            # Record all the shapes
            self.param_shapes = [p.shape for p in self.model_shard.parameters()]

            # Drop all the parameters
            # FIXME: Would it be better to just change device ?
            with torch.no_grad():
                for p in self.model_shard.parameters():
                    p.set_(torch.zeros([0], device=p.device, dtype=p.dtype))

    def forward(self, *inputs):  # type: ignore
        return (self.model_shard(*inputs),) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def forward_load(self) -> None:
        # Resize all the parameter buffers
        if not self.is_owner:
            for p, shape in zip(self.model_shard.parameters(), self.param_shapes):
                if p.numel() == 0:
                    p.set_(torch.zeros(shape, dtype=p.dtype, device=p.device))

        # Fetch or broadcast the latest parameters
        self.backward_load()

    def backward_load(self) -> None:
        # Update all the parameters
        _ = list(
            map(
                lambda x: x.wait(),
                map(
                    lambda p: dist.broadcast(p, self.owner_rank, group=self.process_group, async_op=True),
                    self.model_shard.parameters(),
                ),
            )
        )

    def forward_drop(self) -> None:
        if not self.is_owner:
            for p in self.model_shard.parameters():
                p.grad = None
                # p.set_(torch.zeros([0], device=p.device, dtype=p.dtype))

    def backward_drop(self) -> None:
        if not self.is_owner:
            # Gradients have been reduced and can be discarded
            for p in self.model_shard.parameters():
                p.grad = None
                p.set_(torch.zeros([0], device=p.device, dtype=p.dtype))

    def reduce_grads(self) -> None:
        requests = []
        for p in self.parameters():
            if p.grad is not None:
                p.grad /= self.world_size
                requests.append(dist.reduce(p.grad, dst=self.owner_rank, group=self.process_group, async_op=True))  # type: ignore

        _ = list(map(lambda x: x.wait(), requests))


class ShardSyncLayer(torch.autograd.Function):
    """
     The shard sync layer is a synchronization point between model shards.

     - In the forward pass, it drops parameters in the previous shard and
     loads parameters for the next shard.

     - In the backward pass, it does the reverse and also gathers gradients to the owner.

     It does not change or create any outputs at all, instead it just
     forwards the input as the output.

     NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
     """

    @staticmethod
    def forward(ctx: Any, prev_shard: ModelShard, next_shard: ModelShard, *inputs: Any) -> Any:  # type: ignore
        if prev_shard:
            prev_shard.forward_drop()
        if next_shard:
            next_shard.forward_load()

        ctx.prev_shard = prev_shard
        ctx.next_shard = next_shard

        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        if ctx.next_shard is not None:
            ctx.next_shard.reduce_grads()
            ctx.next_shard.backward_drop()

        if ctx.prev_shard is not None:
            ctx.prev_shard.backward_load()

        # The returned variables need to mirror the forward inputs
        if isinstance(grad_outputs, tuple):
            return None, None, grad_outputs[0]

        return None, None, grad_outputs


class ShardedDataParallelExperimental(nn.Module):
    """Implements distributed data parallel training with optimizer state sharding.

    This experiments with a different way to get to the full zero suite
    The model is sharded, then the normal distributed data parallel algorithm can be used on a per-model shard basis.
    All the gradients are centralized on a given rank (which is model-shard dependent, so that the gradients
    redundancy can be removed).
    Each model shard can be updated by a normal pytorch optimizer.

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
        self.rank = dist.get_rank(self.process_group)
        self.backend = dist.get_backend(group=self.process_group)  # type: ignore

        # Slice the model
        splits = _split(module, self.world_size)

        # Each rank either owns the shard, or temporarily helps processing it in a data parallel fashion
        self.shards: List[nn.Module] = []

        for i_slice, module_shard in enumerate(splits):
            self.shards.append(
                ModelShard(nn.Sequential(*module_shard), owner_rank=i_slice, process_group=self.process_group)
            )

            # Use one normal optimizer per shard
            if i_slice == self.rank:
                self.optimizer = optimizer(nn.Sequential(*module_shard).parameters(), **optimizer_params)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        for i, (prev, next) in enumerate(zip([None, *self.shards], [*self.shards, None])):
            # Per shard FW
            inputs = prev(*inputs) if prev else inputs

            # Call the custom autograd hooks (discard/load shards FW and BW)
            inputs = ShardSyncLayer.apply(prev, next, *inputs)

        return inputs[0] if len(inputs) == 1 else inputs
