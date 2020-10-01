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
from typing import Any, Dict, Generator, List, Type

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
            performing reduce (default: 1M). Used to reduce multiple small
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
        buffer_size: int = 2 ** 20,
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
        self.buffer_size = min(buffer_size, sum(p.numel() for p in self.module.parameters()))
        self._reduce_buffers: Dict[torch.device, torch.Tensor] = {}

        # - One buffer per device
        for per_device in self.sharded_optimizer.per_device_params:
            device = per_device[0][0].device
            dtype = per_device[0][0].dtype
            self._reduce_buffers[device] = torch.empty(self.buffer_size, dtype=dtype, device=device)

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

        if not self.need_reduction or self.accumulate_grads:
            return

        self.need_reduction = False

        with torch.no_grad():
            for per_device in self.sharded_optimizer.per_device_params:
                device = per_device[0][0].device
                self._reduce_grads_task(
                    self._reduce_buffers[device],
                    per_device,
                    group=self.process_group,
                    self_rank=self.rank,
                    world_size=self.world_size,
                )

    @staticmethod
    def _reduce_grads_task(
        buffer: torch.Tensor, params: List[List[Parameter]], group: Any, self_rank: int, world_size: int
    ) -> None:
        """Helper to reduce a list of params. The params are sorted by size, smallest first, which allows for
        an opportunistic bucketing.

        NOTE: All param gradients are assumed to exist"""

        buffer_size = buffer.numel()

        for rank, rank_params in enumerate(params):
            if len(rank_params) == 0:
                continue

            restore_require_grad = []
            requests = []

            # Copy small gradients into per-GPU buffers and then async reduce
            i_p = OSS._bucket(buffer, rank_params, copy=(rank == self_rank), gradients=True)

            if i_p > 0:
                buffer.div_(world_size)  # type: ignore
                requests.append(st.reduce(tensor=buffer, dst=rank, group=group, async_op=True))  # type: ignore

            # Directly reduce the next grads
            for param in rank_params[i_p:]:
                # NOTE: workaround Gloo / leaf variable requiring grad modified in place
                if param.requires_grad:
                    restore_require_grad.append(param)
                    param.requires_grad = False

                param.div_(world_size)  # type: ignore
                requests.append(dist.reduce(tensor=param, dst=rank, group=group, async_op=True))  # type: ignore

            # If applicable, copy back the bucketed values
            OSS._consume_async_work(requests)

            if rank == self_rank:
                OSS._scatter(buffer, rank_params, gradients=True)
            else:
                # Free memory on this rank, these grads are not useful anymore
                for p in rank_params:
                    p.grad = None

            for p in restore_require_grad:
                p.requires_grad = True

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
