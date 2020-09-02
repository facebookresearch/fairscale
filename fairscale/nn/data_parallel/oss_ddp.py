# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A distributed data parallel class that works with OSS optimizer.

Adopted from LegacyDistributedDataParallel module from fairseq.
"""

from collections import OrderedDict
from contextlib import contextmanager
import copy
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, cast

import torch
from torch import nn
import torch.distributed as dist

if TYPE_CHECKING:
    from fairscale.optim import OSS
    from torch import Tensor
    from torch.nn import Parameter
else:
    OSS = Any
    Tensor = Any
    Parameter = Any


class OssDdp(nn.Module):
    """Implements distributed data parallel training with optimizer state sharding.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and does not
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        oss (fairscale.optim.OSS): shared state optimizer
        world_size (int): number of parallel workers
        process_group (optional): the c10d process group to be used for
            distributed gradient reduction. If None, the default WORLD process group
            will be used.
        buffer_size (int, optional): number of elements to buffer before
            performing reduce (default: 256M). Used to reduce multiple small
            params to avoid communication overhead.
    """

    def __init__(
        self, module: nn.Module, oss: OSS, world_size: int, process_group: Any = None, buffer_size: int = 2 ** 28
    ):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.process_group)

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in self.module.parameters()))
        self.buffer: Optional[Tensor] = None

        # Flag used to make sure we only reduce gradients one time in the execution engine
        self.need_reduction = False

        # We can also forcibly accumulate grads locally and only do the
        # gradients-reduce at some later time
        self.accumulate_grads = False

        # TODO (Min): The algorithm here can be improved. We are sorting params by device
        #     and by rank. Then in reduction_fn below, we pack smaller ones into
        #     a buffer for reduction.
        #     We can pre-sort them here and simplify the reduction_fn logic below
        #     since their size shouldn't change.

        # make per-device lists of parameters
        paramlists: OrderedDict = OrderedDict()
        for param in self.module.parameters():
            device = param.device
            if paramlists.get(device) is None:
                paramlists[device] = []
            paramlists[device] += [param]
        self.per_device_params = list(paramlists.values())

        # query oss and build a param-to-rank table
        self.param_rank = {}
        for rank, param_groups in enumerate(oss.partition_parameters()):
            for param_group in param_groups:
                for param in param_group["params"]:
                    self.param_rank[param] = rank

        # sanity checks
        assert len(self.param_rank) == len(list(self.module.parameters())), "number of params do not match"
        for param in self.module.parameters():
            assert param in self.param_rank, f"{param} not in the optimizer"

    def __getstate__(self) -> Dict:
        attrs = copy.copy(self.__dict__)
        return attrs

    def train(self, mode: bool = True) -> "OssDdp":
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
        return self.module(*inputs, **kwargs)

    def reduce(self) -> None:
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """
        assert self.module.training, "Cannot call reduce in eval"

        def reduce_params(params: List[Parameter], params_rank: int) -> None:
            """ Helper to reduce a list of params that should fix in the buffer. """
            assert self.buffer is not None
            buffer: Tensor = cast(Tensor, self.buffer)
            nonzero_buffer = False
            if len(params) > 1:
                offset = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        # The type error could have been fixed in later
                        # version of pytorch. Same elsewhere.
                        buffer[offset : offset + sz].copy_(p.grad.data.view(-1))  # type: ignore
                        nonzero_buffer = True
                    else:
                        buffer[offset : offset + sz].zero_()
                    offset += sz
            else:
                # we only have a single grad to reduce
                p = params[0]
                if p.grad is not None:
                    buffer = p.grad.data
                    nonzero_buffer = True
                elif p.numel() <= self.buffer.numel():
                    buffer = buffer[: p.numel()]
                    buffer.zero_()
                else:
                    buffer = torch.zeros_like(p)

            if nonzero_buffer:
                buffer.div_(self.world_size)  # type: ignore

            dist.reduce(buffer, params_rank, group=self.process_group)  # type: ignore

            if params_rank == self.rank:
                # copy reduced grads back into their original place
                offset = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        p.grad.data.copy_(buffer[offset : offset + sz].view_as(p))  # type: ignore
                    else:
                        p.grad = buffer[offset : offset + sz].view_as(p).clone()
                    offset += sz
            else:
                # zero the grads
                for p in params:
                    if p.grad is not None:
                        p.grad.data.zero_()

        def reduction_fn() -> None:
            # This function only needs to be called once
            if not self.need_reduction or self.accumulate_grads:
                return
            self.need_reduction = False

            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)  # type: ignore

            for params in self.per_device_params:
                # Reduce the gradients in buckets
                offset = 0
                buffered_params: List[Parameter] = []
                param_rank: Optional[int] = None
                for param in params:
                    last_param_rank: Optional[int] = param_rank
                    param_rank = self.param_rank[param]
                    if not param.requires_grad:
                        continue
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    if param.grad.requires_grad:
                        raise RuntimeError("DistributedDataParallel only works with gradients that don't require grad")
                    sz = param.numel()
                    if sz > self.buffer.numel():
                        # reduce big params directly
                        assert param_rank is not None
                        reduce_params([param], cast(int, param_rank))
                    else:
                        # smaller params are packed together from the same device
                        # and same rank.
                        if offset + sz > self.buffer.numel() or (
                            last_param_rank is not None and last_param_rank != param_rank
                        ):
                            assert last_param_rank is not None
                            reduce_params(buffered_params, cast(int, last_param_rank))
                            offset = 0
                            buffered_params.clear()
                        buffered_params.append(param)
                        offset += sz

                if len(buffered_params) > 0:
                    assert param_rank is not None
                    reduce_params(buffered_params, cast(int, param_rank))

        reduction_fn()
