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
from typing import Any, Dict, Generator, List, Optional, Type, cast

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim import OSS


class ShardedDataParallel(nn.Module):
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
            performing reduce (default: 256M). Used to reduce multiple small
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
        buffer_size: int = 2 ** 28,
    ):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.process_group)
        self.broadcast_buffers = broadcast_buffers
        self.authoritative_rank = 0

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in self.module.parameters()))
        self.buffer: Optional[Tensor] = None

        # Flag used to make sure we only reduce gradients one time in the execution engine
        self.need_reduction = False

        # We can also forcibly accumulate grads locally and only do the
        # gradients-reduce at some later time
        self.accumulate_grads = False

        # Build the sharded optimizer
        self.sharded_optimizer = OSS(self.module.parameters(), optim=optimizer, group=process_group, **optimizer_params)

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

    def train(self, mode: bool = True) -> "ShardedDataParallel":
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

        def reduce_grads(params: List[Parameter], params_rank: int) -> None:
            """ Helper to reduce a list of params that should fit in the buffer.
            NOTE: All param gradients are assumed to exist"""
            assert self.buffer is not None

            # Fill in the packed IO buffer
            buffer: Tensor = cast(Tensor, self.buffer)
            if len(params) > 1:
                offset = 0
                for p in params:
                    sz = p.numel()
                    buffer[offset : offset + sz].copy_(p.grad.data.view(-1))  # type: ignore
                    offset += sz
            else:
                # we only have a single grad to reduce
                buffer = params[0].grad.data  # type: ignore

            # Reduce
            buffer.div_(self.world_size)  # type: ignore
            dist.reduce(tensor=buffer, dst=params_rank, group=self.process_group)  # type: ignore

            # Copy reduced grads back into their original place, or free corresponding memory
            if params_rank == self.rank:
                offset = 0
                for p in params:
                    sz = p.numel()
                    p.grad.data.copy_(buffer[offset : offset + sz].view_as(p))  # type: ignore
                    offset += sz
            else:
                for p in params:
                    p.grad = None

        def reduction_fn() -> None:
            # This function only needs to be called once
            if not self.need_reduction or self.accumulate_grads:
                return
            self.need_reduction = False

            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)  # type: ignore

            for per_device in self.sharded_optimizer.per_device_params:
                for rank, params in enumerate(per_device):

                    # Reduce the gradients in buckets
                    offset = 0
                    buffered_params: List[Parameter] = []
                    for param in params:
                        if not param.requires_grad:
                            continue

                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        if param.grad.requires_grad:
                            raise RuntimeError(
                                "DistributedDataParallel only works with gradients that don't require grad"
                            )
                        sz = param.numel()
                        if sz > self.buffer.numel():
                            # reduce big params directly
                            reduce_grads([param], cast(int, rank))
                        else:
                            # smaller params are packed together from the same device
                            # and same rank.
                            if offset + sz > self.buffer.numel():
                                reduce_grads(buffered_params, cast(int, rank))
                                offset = 0
                                buffered_params.clear()
                            buffered_params.append(cast(Parameter, param))
                            offset += sz

                    if len(buffered_params) > 0:
                        reduce_grads(buffered_params, rank)

        reduction_fn()

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
