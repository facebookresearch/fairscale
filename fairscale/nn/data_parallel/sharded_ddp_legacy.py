# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A distributed data parallel class that works with OSS optimizer.

Adopted from LegacyDistributedDataParallel module from fairseq.
"""

from contextlib import contextmanager
import copy
from typing import Any, Dict, Generator, List, Type, cast

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim import OSS


class ShardedDataParallelLegacy(nn.Module):
    """Implements distributed data parallel training with optimizer state sharding.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and optionally
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        optimizer (~torch.optim.Optimizer): optimizer to be used for training
        optimizer_params(Dict): extra parameters for the optimizer
        world_size (int): number of parallel workers
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
        the module at beginning of the forward function. (default: ``True``)
        process_group (optional): the c10d process group to be used for
            distributed gradient reduction. If None, the default WORLD process group
            will be used.
        buffer_size (int, optional): number of elements to buffer before
            performing reduce (default: 512k). Used to reduce multiple small
            params to avoid communication overhead.
    """

    def __init__(
        self,
        module: nn.Module,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict[str, Any],
        world_size: int,
        broadcast_buffers: bool,
        process_group: Any = None,
        buffer_size: int = 2 ** 19,
    ):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.process_group)
        self.broadcast_buffers = broadcast_buffers
        self.authoritative_rank = 0

        # Flag used to make sure we only reduce gradients one time in the execution engine
        self.need_reduction = False

        # We can also forcibly accumulate grads locally and only do the
        # gradients-reduce at some later time
        self.accumulate_grads = False

        # Build the sharded optimizer
        self.sharded_optimizer = OSS(self.module.parameters(), optim=optimizer, group=process_group, **optimizer_params)

        # Allocate reduce buffers
        # - Never use a bigger buffer than the number of model params
        buffer_size = min(buffer_size, sum(p.numel() for p in self.module.parameters()))
        self._reduce_buffers: Dict[torch.device, List[torch.Tensor]] = {}

        # - One buffer per rank per device
        for device, per_device in self.sharded_optimizer.per_device_params.items():
            buffer_dtype = per_device[0][0].dtype
            self._reduce_buffers[device] = [
                torch.zeros(buffer_size, dtype=buffer_dtype, device=device) for _ in range(len(per_device))
            ]

        # Sanity checks
        assert len(self.sharded_optimizer.param_to_rank) == len(
            list(self.module.parameters())
        ), "number of params do not match"
        for param in self.module.parameters():
            assert param in self.sharded_optimizer.param_to_rank, f"{param} not in the optimizer"

    def __getstate__(self) -> Dict:
        attrs = copy.copy(self.__dict__)
        return attrs

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.sharded_optimizer

    def train(self, mode: bool = True) -> "ShardedDataParallelLegacy":
        pre_mode = self.module.training
        self.module.train(mode)
        if self.module.training:
            assert not self.need_reduction or pre_mode, "incorrect state transition"
        else:
            assert not self.need_reduction, "try to enter eval with grads unreduced"
        return self

    @contextmanager
    def no_sync(self) -> Generator:
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

    def forward(self, *inputs: Any, **kwargs: Any) -> Tensor:
        if self.module.training:
            if self.need_reduction:
                raise RuntimeError("OssDdp requires explicit reduction, must call OssDdp.reduce")
            if not self.accumulate_grads:
                self.need_reduction = True
            if self.broadcast_buffers and len(list(self.module.buffers())) > 0:
                self._sync_buffers()

        return self.module(*inputs, **kwargs)

    def reduce(self) -> None:
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """
        assert self.module.training, "Cannot call reduce in eval"

        if not self.need_reduction or self.accumulate_grads:
            return

        self.need_reduction = False

        with torch.no_grad():
            for device, per_device in self.sharded_optimizer.per_device_params.items():
                self._reduce_grads_task(
                    self._reduce_buffers[device],
                    per_device,
                    group=self.process_group,
                    self_rank=self.rank,
                    world_size=self.world_size,
                )

    @staticmethod
    def _reduce_grads_task(
        buffers: List[torch.Tensor], per_rank_params: List[List[Parameter]], group: Any, self_rank: int, world_size: int
    ) -> None:
        """Helper to reduce a list of params. The params are sorted by size, smallest first, which allows for
        an opportunistic bucketing.

        NOTE: All param gradients are assumed to exist"""

        buffer_size = buffers[0].numel()
        bucket_requests = []
        requests = []

        for (rank, params), buffer in zip(enumerate(per_rank_params), buffers):
            # All the params are sorted per rank and per increasing size
            if len(params) == 0:
                continue

            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            global_rank = OSS.get_global_rank(group, rank)

            # Copy small gradients into per-GPU buffers and then async reduce
            i_bucketed = 0  # the number of tensors packed in the buffer
            offset = 0

            # Since all the parameters are already sorted per increasing size, we only need to consider the first ones.
            while i_bucketed < len(params) and offset + params[i_bucketed].numel() < buffer_size:
                end = offset + params[i_bucketed].numel()
                buffer[offset:end].copy_(params[i_bucketed].grad.data.view(-1))  # type: ignore
                offset = end
                i_bucketed += 1

            if i_bucketed > 0:
                buffer.div_(world_size)  # type: ignore
                bucket_requests.append(
                    (
                        dist.reduce(tensor=buffer, dst=global_rank, group=group, async_op=True),  # type: ignore
                        rank,
                    )
                )

            # Directly reduce the other grads
            for p in params[i_bucketed:]:
                p.grad = cast(Tensor, p.grad)
                if p.grad.requires_grad:
                    raise RuntimeError("DistributedDataParallel only works with gradients that don't require grad")

                p.grad.div_(world_size)  # type: ignore
                requests.append(dist.reduce(tensor=p.grad, dst=global_rank, group=group, async_op=True))  # type: ignore

        # Unroll the initial packed small gradients, as soon as possible
        for future, rank in bucket_requests:
            future.wait()

            if rank == self_rank:
                i_bucketed = 0  # the number of tensors packed in the buffer
                offset = 0
                params = per_rank_params[rank]
                buffer = buffers[rank]

                while i_bucketed < len(params) and offset + params[i_bucketed].numel() < buffer_size:
                    end = offset + params[i_bucketed].numel()
                    params[i_bucketed].grad.data.copy_(buffer[offset:end].view_as(params[i_bucketed]))  # type: ignore
                    offset = end
                    i_bucketed += 1

        # Make sure that we're done with this device before moving on and cleaning the unused params
        _ = list(map(lambda x: x.wait(), requests))

    def _sync_buffers(self) -> None:
        """
        Sync all the param buffers in between ranks.
        TODO: Could be worth bucketing ?
        """
        _ = list(
            map(
                lambda x: x.wait(),
                map(
                    lambda x: dist.broadcast(x, self.authoritative_rank, self.process_group, async_op=True),
                    self.module.buffers(),
                ),
            )
        )
