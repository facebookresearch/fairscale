# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A distributed data parallel class that shards the model and the optimizer into pieces.

See https://github.com/pytorch/pytorch/issues/42849 for more context. Credits to Shen Li for the original idea

"""

from builtins import isinstance
import logging
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
import torch.distributed as dist


def _split(modules: nn.Sequential, number_shards: int) -> List[List[nn.Module]]:
    splits: List[List[nn.Module]] = [[] for _ in range(number_shards)]
    n = len(modules) // number_shards

    # Count the number of parameters per exposed layer, use that as a proxy for memory footprint
    number_parameters_per_layer = [sum(p.numel() for p in m.parameters()) for m in modules]
    total_number_params = sum(number_parameters_per_layer)
    number_parameters_per_shard = total_number_params // number_shards

    logging.info(
        f"This model has {total_number_params/1e6:.2f}M parameters, aiming for {number_parameters_per_shard/1e6:.2f}M parameters per shard"
    )

    current_shard = 0

    for m in modules:
        # Number of parameters in the current shard
        current_shard_params = sum(p.numel() for sm in splits[current_shard] for p in sm.parameters())

        # This shard is big enough, point to the next one
        if (
            current_shard_params + sum(p.numel() for p in m.parameters()) > number_parameters_per_shard
            and current_shard < number_shards - 1
        ):
            current_shard += 1

        splits[current_shard].append(m)

    return splits


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the fly for the FW pass and gather gradients.
    Depending on whether this rank is or is not the `owner_rank`, this ModelShard either only handles
    a shard of the compute and is stateless or also owns the up to date state.
    """

    def __init__(
        self,
        cpu_model_shard: nn.Module,
        owner_rank: int,
        process_group: Any,
        device: torch.device,
        offload_device: torch.device,
        broadcast_bufers: bool = True,
        offload_optimizer: bool = False,
    ):
        super().__init__()
        self.owner_rank = owner_rank
        self.process_group = process_group
        self.model_shard = cpu_model_shard

        self.rank = OffloadDataParallelExperimental.get_global_rank(
            self.process_group, dist.get_rank(self.process_group)
        )
        self.is_owner = self.rank == self.owner_rank
        self.world_size = dist.get_world_size(self.process_group)
        self.broadcast_buffers = broadcast_bufers

        # Save all the parameter sizes to be able to restore them
        self.device = device
        self.offload_device = offload_device
        self.should_offload_optimizer = offload_optimizer

        self.model_shard.to(offload_device)

        if not self.is_owner:
            # Record all the shapes
            self.param_shapes = [p.shape for p in self.model_shard.parameters()]

    def forward(self, *inputs):  # type: ignore
        if self.broadcast_buffers and len(list(self.model_shard.buffers())) > 0:
            self.sync_buffers(non_blocking=False)

        return (self.model_shard(*inputs),) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def to(self, device: torch.device) -> "ModelShard":  # type: ignore
        # Make sure that the lookahead and lookback shards are not captured by this call
        self.model_shard.to(device)
        return self

    def train(self, mode: bool = True) -> "ModelShard":
        # Make sure that the lookahead and lookback shards are not captured by this call
        self.model_shard.train(mode)
        return self

    def to_device(self) -> None:
        self.model_shard.to(device=self.device, non_blocking=True)

    def forward_load(self, sync: bool = False, non_blocking: bool = True) -> Optional[List[Any]]:
        # Restore all the parameter buffers
        self.model_shard.to(device=self.device, non_blocking=non_blocking)

        # Fetch or broadcast the latest parameters
        if sync:
            requests = list(
                map(
                    lambda p: dist.broadcast(p, self.owner_rank, group=self.process_group, async_op=True),
                    self.model_shard.parameters(),
                ),
            )
            return requests if non_blocking else self._sync(requests)

        return None

    def backward_load(self, non_blocking: bool = True) -> None:
        self.model_shard.to(self.device, non_blocking=non_blocking)

    def forward_drop(self, non_blocking: bool = True) -> None:
        for p in self.model_shard.parameters():
            p.grad = None

        self.model_shard.to(self.offload_device, non_blocking=non_blocking)

    def backward_drop(self, non_blocking: bool = True) -> None:
        if not self.is_owner:
            # Gradients have been reduced and can be discarded
            for p in self.model_shard.parameters():
                p.grad = None

        if self.should_offload_optimizer or not self.is_owner:
            # Either the optimization takes place on the offload device
            # or this rank does not own this shard
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)

    def reduce_grads(self, non_blocking: bool) -> Optional[List[Any]]:
        requests = []

        # Send all the gradients to the owner
        for p in filter(lambda p: p.grad is not None, self.parameters()):
            assert p.grad is not None  # useless but mypy requires that

            p.grad /= self.world_size
            requests.append(dist.reduce(p.grad.data, dst=self.owner_rank, group=self.process_group, async_op=True))

        return requests if non_blocking else self._sync(requests)

    def sync_buffers(self, non_blocking: bool = True) -> Optional[List[Any]]:
        """
        Sync all the param buffers in between ranks.
        """
        requests = list(
            map(
                lambda x: dist.broadcast(x, self.owner_rank, self.process_group, async_op=True),
                self.model_shard.buffers(),
            ),
        )
        return requests if non_blocking else self._sync(requests)

    def sync_parameters(self, non_blocking: bool = True) -> Optional[List[Any]]:
        """
        Sync all the parameters in between ranks.
        """
        requests = list(
            map(
                lambda x: dist.broadcast(x.data, self.owner_rank, self.process_group, async_op=True),
                self.model_shard.parameters(),
            ),
        )
        return requests if non_blocking else self._sync(requests)

    @staticmethod
    def _sync(requests: Optional[List[Any]]) -> None:
        """
        Make an async function synchronous.
        Use this to wrap the function call directly
        """
        if requests:
            _ = list(map(lambda x: x.wait(), requests))
        return


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
    def forward(ctx: Any, p2_shard: Optional[ModelShard], p1_shard: Optional[ModelShard], n1_shard: Optional[ModelShard], n2_shard: Optional[ModelShard], *inputs: Any) -> Any:  # type: ignore
        # Drop the shard we just went through, except if this is the last one in line
        if p1_shard and n1_shard:
            p1_shard.forward_drop(non_blocking=True)

        # Start the load of the next shard in line, opportunistically look ahead
        if n2_shard:
            n2_shard.forward_load(sync=True, non_blocking=True)

        ctx.p2_shard = p2_shard
        ctx.p1_shard = p1_shard
        ctx.n1_shard = n1_shard
        ctx.n2_shard = n2_shard

        # FIXME: handle corner cases / type dependent
        outputs = inputs

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        if ctx.n1_shard:
            ctx.n1_shard.reduce_grads(non_blocking=True)
            ctx.n1_shard.backward_drop(non_blocking=True)

        # Opportunistically pre-load ahead of the compute wavefront
        if ctx.p2_shard:
            ctx.p2_shard.backward_load(non_blocking=True)

        # The returned variables need to mirror the forward inputs
        if isinstance(grad_outputs, tuple):
            return None, None, None, None, grad_outputs[0]

        return None, None, None, None, grad_outputs


class OffloadDataParallelExperimental(nn.Module):
    """Implements distributed data parallel training with optimizer state sharding and model sharding.

    This experiments with a different way to get to the full zero suite
    The model is sharded, then the normal distributed data parallel algorithm can be used on a per-model shard basis.
    Each shard is offloaded and loaded following a compute wavefront, during the forward and backward pass.

    All the gradients are centralized on a given rank (which is model-shard dependent, so that the gradients
    redundancy can be removed). Each model shard can be updated by a normal pytorch optimizer.

    Args:
        module (~torch.nn.Sequential): module to be parallelized
        optimizer (~torch.optim.Optimizer): optimizer to be used for training
        optimizer_params(Dict): extra parameters for the optimizer
        world_size (int): number of parallel workers

        device (torch.device):
            device where the active model should reside

        offload_device (torch.device):
            device where the inactive model should reside

        process_group (optional):
            the c10d process group to be used for
            distributed gradient reduction. If None, the default WORLD process group
            will be used.

        broadcast_buffers (bool, optional):
            whether to sync all the model buffers at the beginning of a FW pass

        offload_optimizer (bool):
            move optimizer computations to the offload device (for instance CPU), which saves more memory but slows down
    """

    def __init__(
        self,
        model_cpu: nn.Sequential,  # hard pre-requisite for now, easier model slicing
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict[str, Any],
        world_size: int,
        device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
        process_group: Any = None,
        broadcast_buffers: bool = True,
        offload_optimizer: bool = False,
    ):
        super().__init__()

        self.world_size = world_size
        self.process_group = process_group if process_group is not None else torch.distributed.group.WORLD
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = self.get_global_rank(self.process_group, self.rank)
        self.backend = dist.get_backend(group=self.process_group)
        self.device = device
        self.offload_device = device

        # Slice the model into roughly equivalent sequential shards
        splits = _split(model_cpu, self.world_size)

        # Each rank either owns the slice, or temporarily helps processing it in a data parallel fashion
        self.model_slices: List[nn.Module] = []

        for i_slice, module_shard in enumerate(splits):
            global_owner_rank = self.get_global_rank(self.process_group, i_slice)

            # Add one dataparallel model handling this slice
            self.model_slices.append(
                ModelShard(
                    cpu_model_shard=nn.Sequential(*module_shard),
                    owner_rank=global_owner_rank,
                    process_group=self.process_group,
                    device=device,
                    offload_device=offload_device,
                    broadcast_bufers=broadcast_buffers,
                    offload_optimizer=False,
                )
            )

            # Use one normal optimizer per shard
            if i_slice == self.rank:
                self.optimizer = optimizer(nn.Sequential(*module_shard).parameters(), **optimizer_params)

        # Expose a unified view of the slices
        self.model = torch.nn.Sequential(*self.model_slices)
        self.sync_ranks()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        # Slice per slice FW, sync in between
        syncRanks = ShardSyncLayer.apply

        for i, (p2, p1, n1, n2) in enumerate(
            zip(
                [None, None, *self.model_slices],
                [None, *self.model_slices],
                [*self.model_slices, None],
                [*self.model_slices, None, None],
            )
        ):
            # Per shard FW
            inputs = p1(*inputs) if p1 else inputs

            # Call the custom autograd hooks (discard/load slices FW and BW)
            inputs = syncRanks(p2, p1, n1, n2, *inputs)

        return inputs[0] if len(inputs) == 1 else inputs

    @staticmethod
    def get_global_rank(group: Any, rank: int) -> int:
        if group is dist.group.WORLD:
            return rank
        else:
            global_rank = dist.distributed_c10d._get_global_rank(group, rank)
        return global_rank

    def sync_ranks(self, non_blocking: bool = False) -> None:
        for model_slice in self.model_slices:
            if self.backend != "nccl":
                model_slice.sync_parameters(non_blocking=non_blocking)  # type: ignore
                model_slice.sync_buffers(non_blocking=non_blocking)  # type: ignore
            else:
                # NCCL requires the tensors to be on GPU for broadcast
                model_slice.to(self.device)
                model_slice.sync_parameters(non_blocking=False)  # type: ignore
                model_slice.sync_buffers(non_blocking=False)  # type: ignore
                model_slice.to("cpu")
