# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

if torch.__version__.split(".")[:2] >= ["1", "8"]:
    from torch.distributed import all_reduce
else:
    # Copied from https://github.com/pytorch/pytorch/blob/v1.8.1/torch/distributed/nn/functional.py
    class _AllReduce(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op, group, tensor):  # type: ignore
            ctx.group = group
            ctx.op = op
            tensor = tensor.clone()
            dist.all_reduce(tensor, op=op, group=group)
            return tensor

        @staticmethod
        def backward(ctx, grad_output):  # type: ignore
            return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD):  # type: ignore
        return _AllReduce.apply(op, group, tensor)


def _forward(
    input: torch.Tensor,
    affine: bool,
    track_running_stats: bool,
    mean: torch.Tensor,
    var: torch.Tensor,
    momentum: float,
    eps: float,
    weight_param: torch.Tensor,
    bias_param: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    total_count: torch.Tensor,
) -> torch.Tensor:
    scale = torch.rsqrt(var + eps)
    bias = -mean * scale
    if affine:
        scale = scale * weight_param.reshape(mean.shape)
        bias = bias + bias_param.reshape(mean.shape)

    if track_running_stats:
        with torch.no_grad():
            unbiased_var = var * (total_count / (total_count - 1))
            running_mean += momentum * (mean.reshape(-1) - running_mean)
            running_var += momentum * (unbiased_var.reshape(-1) - running_var)

    return input * scale + bias


if torch.__version__.split(".")[:2] >= ["1", "7"]:
    _forward = torch.jit.script(_forward)  # type: ignore


class SyncBatchNorm(torch.nn.BatchNorm2d):
    """
    Fast re-implementation of ``torch.nn.SyncBatchNorm`` that can achieve a speedup
    of 5x or more over the default implementation depending on size of the input
    and number of distributed world.
    """

    def __init__(
        self, *args: Tuple[Any, ...], process_group: Optional[ProcessGroup] = None, **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self._process_group = process_group if process_group is not None else dist.group.WORLD

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not dist.is_initialized() or not self.training:
            return super().forward(input)

        dim = [d for d in range(input.ndim) if d != 1]
        count = torch.full((1,), input.numel() // input.size(1), device=input.device, dtype=input.dtype)
        total_count = count.clone()
        handle = all_reduce(total_count, group=self._process_group, async_op=True)
        mean = torch.mean(input, dim=dim, keepdim=True)
        meansqr = torch.mean(input * input, dim=dim, keepdim=True)
        vec = torch.cat([mean, meansqr])
        handle.wait()
        vec = vec * (count / total_count)
        all_reduce(vec, group=self._process_group)
        mean, meansqr = vec.chunk(2)
        var = meansqr - mean * mean

        return _forward(
            input,
            self.affine,
            self.track_running_stats,
            mean,
            var,
            self.momentum,
            self.eps,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            total_count,
        )

    @classmethod
    def convert_sync_batchnorm(
        cls, module: torch.nn.Module, process_group: Optional[ProcessGroup] = None
    ) -> torch.nn.Module:
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`fairscale.experimental.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = fairscale.experimental.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,  # type: ignore
                module.eps,  # type: ignore
                module.momentum,  # type: ignore
                module.affine,  # type: ignore
                module.track_running_stats,  # type: ignore
                process_group=process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output
