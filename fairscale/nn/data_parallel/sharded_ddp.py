# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A module wrapper to go with a Sharded Optimizer in order to handle targeted gradient reduction/gathering automatically.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Union, cast

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim.oss import OSS


def _get_global_rank(group: Any, rank: int) -> int:
    if group is dist.group.WORLD:
        return rank
    else:
        global_rank = dist.distributed_c10d._get_global_rank(group, rank)  # type: ignore
    return global_rank


class ModelDispatch(nn.Module):
    """
    Wrap a model, make it possible to load parameters on the fly for the FW pass and gather gradients.
    Depending on whether this rank is or is not the `owner_rank`, this ModelShard either only handles
    a shard of the compute and is stateless or also owns the up to date state.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any,
        broadcast_buffers: bool = True,
        reference_rank: int = 0,
        buffer_size: int = 2 ** 19,
    ):
        super().__init__()
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.base_model = base_model

        self.sharded_optimizers = (
            [cast(OSS, sharded_optimizer)] if isinstance(sharded_optimizer, OSS) else sharded_optimizer
        )
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = _get_global_rank(self.process_group, dist.get_rank(self.process_group))
        self.reference_global_rank = _get_global_rank(self.process_group, reference_rank)
        self.world_size = dist.get_world_size(self.process_group)
        self.broadcast_model_buffers = broadcast_buffers and len(list(self.base_model.buffers(recurse=True))) > 0
        self.backend = dist.get_backend()

        # Allocate reduce buffers
        # - Never use a bigger buffer than the number of model params
        buffer_size = min(buffer_size, sum(p.numel() for p in self.base_model.parameters()))
        self._reduce_buffers: Dict[OSS, Dict[torch.device, List[torch.Tensor]]] = defaultdict(dict)

        # - One buffer per rank per device for each optimizer
        for sharded_optimizer in self.sharded_optimizers:
            for device, per_device in sharded_optimizer.per_device_params.items():
                buffer_dtype = per_device[0][0].dtype
                self._reduce_buffers[sharded_optimizer][device] = [
                    torch.zeros(buffer_size, dtype=buffer_dtype, device=device) for _ in range(len(per_device))
                ]

        # Sync all the ranks
        self.sync_all_params()

    def forward(self, *inputs):  # type: ignore
        if self.broadcast_model_buffers:
            self.sync_buffers(non_blocking=False)

        return (self.base_model(*inputs),) if isinstance(inputs, tuple) else self.base_model(inputs)

    def dispatch_grads(self) -> None:
        """
        Reduce -NOTE: could become gather- all the gradients to the appropriate ranks
        """

        with torch.no_grad():
            # Make sure that all ranks are done
            torch.distributed.barrier()

            for sharded_optimizer in self.sharded_optimizers:
                for device, per_device in sharded_optimizer.per_device_params.items():
                    # Reduce all params to appropriate ranks
                    self._reduce_grads_task(
                        self._reduce_buffers[sharded_optimizer][device],
                        per_device,
                        group=self.process_group,
                        self_rank=self.rank,
                        world_size=self.world_size,
                    )

    @staticmethod
    def _reduce_grads_task(
        buffers: List[torch.Tensor],
        per_rank_params: List[List[Parameter]],
        group: Any,
        self_rank: int,
        world_size: int,
    ) -> None:
        """Helper to reduce a list of params. The params are sorted by size, smallest first, which allows for
        an opportunistic bucketing.

        .. warning: All param gradients are assumed to exist
        .. warning: Reduced grads are removed from the ranks which don't own them, to save memory"""

        buffer_size = buffers[0].numel()
        direct_requests = []
        bucket_requests = []

        _world_size = float(world_size)

        # First issue all the reduce requests, for all devices, and collect the pseudo-futures. Two parts:
        #  - the smallest gradients are bucketed
        #  - the biggest are reduced directly
        for (dst_rank, params), buffer in zip(enumerate(per_rank_params), buffers):
            global_dst_rank = OSS.get_global_rank(group, dst_rank)

            # Copy small gradients into per-GPU buffers and then async reduce
            offset = 0
            bucket_sent = False
            bucket_params = []

            # All the params are sorted per rank and per increasing size
            for p in params:
                # Since all the parameters are already sorted per increasing size, we only need to consider the first ones.
                if not bucket_sent and offset + p.numel() < buffer_size:
                    end = offset + p.numel()
                    buffer[offset:end].copy_(p.grad.data.view(-1))  # type: ignore
                    bucket_params.append((p, offset, end))

                    offset = end
                    if dst_rank != self_rank:
                        # This rank is not the owner, these gradients have been copied and can be released
                        p.grad = None
                else:
                    if offset > 0 and not bucket_sent:
                        # Bucket is full, send asap
                        buffer.div_(_world_size)
                        bucket_requests.append(
                            (
                                dist.reduce(tensor=buffer, dst=global_dst_rank, group=group, async_op=True),  # type: ignore
                                dst_rank,
                                bucket_params,
                            )
                        )

                        bucket_sent = True

                    # Directly reduce the other grads
                    p.grad = cast(Parameter, p.grad)
                    if p.grad.requires_grad:
                        raise RuntimeError("DistributedDataParallel only works with gradients that don't require grad")

                    p.grad.data.div_(_world_size)
                    direct_requests.append(
                        (dist.reduce(tensor=p.grad.data, dst=global_dst_rank, group=group, async_op=True), dst_rank, p,)  # type: ignore
                    )

            # Catch a trailing bucket
            if not bucket_sent:
                buffer.div_(_world_size)
                bucket_requests.append(
                    (
                        dist.reduce(tensor=buffer, dst=global_dst_rank, group=group, async_op=True),  # type: ignore
                        dst_rank,
                        bucket_params,
                    )
                )

        # Now unroll the bucketed small gradients
        for work_handle, dst_rank, bucket_params in bucket_requests:
            work_handle.wait()

            if dst_rank == self_rank:
                for p, offset, end in bucket_params:
                    p.grad.data.copy_(buffers[dst_rank][offset:end].view_as(p.data))  # type: ignore

        # Finally, make sure that we're done with this device before moving on and cleaning the unused params
        for work_handle, dst_rank, param in direct_requests:
            work_handle.wait()

            if dst_rank != self_rank:
                # This gradient has been reduced and this rank is not the owner, it can be released
                param.grad = None

    def sync_buffers(self, non_blocking: bool = False) -> Optional[List[Any]]:
        """
        Sync all the param buffers in between ranks.
        TODO: Could be worth bucketing ?
        """

        work_handles = [
            dist.broadcast(x.data, self.reference_global_rank, self.process_group, async_op=True)
            for x in self.base_model.buffers(recurse=True)
        ]
        return work_handles if non_blocking else self.wait(work_handles)

    def sync_all_params(self) -> None:
        """
        Sync the complete model states in between the ranks
        """
        work_handles = [
            dist.broadcast(t, src=self.reference_global_rank, group=self.process_group, async_op=True)
            for t in self.base_model.state_dict().values()
        ]
        self.wait(work_handles)

    @staticmethod
    def wait(requests: Optional[List[Any]]) -> None:
        """
        Make an async function synchronous.
        Use this to wrap the function call directly
        """
        if requests:
            _ = list(map(lambda x: x.wait(), requests))
        return


class DispatchLayer(torch.autograd.Function):
    """
     The dispatch layer is a synchronization point between model shards.

     - In the forward pass it does nothing
     - In the backward pass, it gathers gradients to the owner.

     NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
     """

    @staticmethod
    def forward(ctx: Any, model: ModelDispatch, *inputs: Any) -> Any:  # type: ignore
        # Store a handle to the model for the BW dispatch
        ctx.model = model
        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        ctx.model.dispatch_grads()
        return tuple([None, *grad_outputs])


class ShardedDataParallel(nn.Module):
    """
    Wrap the model, and reduce the gradients to the right rank after the backward pass.

    - the partition is given by the sharded optimizer
    - wrap the base model with a model which knows where to reduce each gradient
    - add an autograd function which calls the model grad dispatch on the way back

     Args:
        base_model (nn.Module):
            model to be wrapped
        sharded_optimizer (OSS, or list of OSS):
            the sharded optimizer(s) which will decide the gradient partitioning
    Keyword Args:
        process_group (torch.nn.Optimizer):
            optimizer to shard (default: SGD)
        process_group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffers (bool):
            whether to broadcast model buffers in between ranks at the beginning of each forward pass
        buffer_size (int):
            the size of the buffer in bits used to batch the small parameter tensors (default 128k).
    """

    def __init__(
        self,
        base_model: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any = None,
        broadcast_buffers: bool = True,
        buffer_size: int = 2 ** 17,
    ):
        super().__init__()

        self.model_dispatch = ModelDispatch(
            base_model=base_model,
            sharded_optimizer=sharded_optimizer,
            process_group=process_group,
            broadcast_buffers=broadcast_buffers,
            reference_rank=0,
            buffer_size=buffer_size,
        )

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        # All inputs need to required_grad for autograd to properly track the first dispatch layer
        if isinstance(inputs, tuple):
            for i in inputs:
                i.requires_grad = True
        elif isinstance(inputs, torch.Tensor):
            inputs.requires_grad = True

        # Register the model dispatch in the autograd graph
        inputs = DispatchLayer.apply(self.model_dispatch, *inputs)

        # Normal model FW
        outputs = self.model_dispatch(*inputs)

        return outputs[0] if len(outputs) == 1 else outputs

    def reduce(self) -> None:
        logging.warning("This is not useful anymore, gradients have been reduced automatically with the backward pass")
