# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _forward(
    input: torch.Tensor,
    affine: bool,
    track_running_stats: bool,
    mean: torch.Tensor,
    var: torch.Tensor,
    invstd: torch.Tensor,
    momentum: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    total_count: torch.Tensor,
) -> torch.Tensor:
    if track_running_stats:
        with torch.no_grad():
            unbiased_var = var * (total_count / (total_count - 1))
            running_mean += momentum * (mean.reshape(-1) - running_mean)
            running_var += momentum * (unbiased_var.reshape(-1) - running_var)

    if affine:
        return (input - mean) * (invstd * weight.reshape_as(mean)) + bias.reshape_as(mean)
    else:
        return (input - mean) * invstd


if torch.__version__.split(".")[:2] >= ["1", "7"]:
    _forward = torch.jit.script(_forward)  # type: ignore


class _SyncBatchNormFunction(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(
        ctx, input, weight, bias, affine, track_running_stats, running_mean, running_var, eps, momentum, process_group
    ):
        dim = [d for d in range(input.ndim) if d != 1]
        count = torch.full((1,), input.numel() // input.size(1), device=input.device, dtype=input.dtype)
        total_count = count.clone()
        all_reduce_handle = dist.all_reduce(total_count, group=process_group, async_op=True)
        mean = torch.mean(input, dim=dim, keepdim=True)
        meansqr = torch.mean(input * input, dim=dim, keepdim=True)
        vec = torch.cat([mean, meansqr])
        all_reduce_handle.wait()
        vec = vec * (count / total_count)
        dist.all_reduce(vec, group=process_group)
        mean, meansqr = vec.chunk(2)
        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + eps)

        ctx.save_for_backward(input, weight, bias, mean, invstd, total_count)
        ctx.process_group = process_group

        return _forward(
            input,
            affine,
            track_running_stats,
            mean,
            var,
            invstd,
            momentum,
            weight,
            bias,
            running_mean,
            running_var,
            total_count,
        )

    @staticmethod
    # type: ignore
    def backward(ctx, grad_output):
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]

        grad_input = None
        grad_weight = None
        grad_bias = None

        input, weight, bias, mean, invstd, total_count = ctx.saved_tensors
        process_group = ctx.process_group

        dim = [d for d in range(input.ndim) if d != 1]
        if needs_input_grad or needs_weight_grad:
            grad_common = torch.sum(
                (input - mean) * grad_output, dim=dim, keepdim=True
            )  # common to grad_weight and grad_invstd

        if needs_input_grad:
            if weight is None:  # i.e. affine is False
                grad_input = invstd * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common
            else:
                grad_input = (invstd * weight.reshape_as(mean)) * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common * weight.reshape_as(mean)
            grad_var = -0.5 * invstd.pow(3) * grad_invstd
            grad_mean += -2 * mean * grad_var
            grad_meansqr = grad_var
            vec = torch.cat([grad_mean, grad_meansqr])
            all_reduce_handle = dist.all_reduce(vec, group=process_group, async_op=True)

        if needs_weight_grad:
            grad_weight = (grad_common * invstd).resize_as(weight)
            grad_bias = torch.sum(grad_output, dim=dim)

        if needs_input_grad:
            all_reduce_handle.wait()
            vec = vec / total_count  # NOTE(msb) removed '* count' here to avoid '/  count' below
            grad_mean, grad_meansqr = vec.chunk(2)
            grad_input += grad_mean  # removed '/ count'
            grad_input += input * (2 * grad_meansqr)  # removed '/ count'

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class SyncBatchNorm(torch.nn.BatchNorm2d):
    """
    Fast re-implementation of ``torch.nn.SyncBatchNorm`` that can achieve a speedup
    of 5x or more over the default implementation depending on size of the input
    and number of distributed workers.
    """

    def __init__(
        self, *args: Tuple[Any, ...], process_group: Optional[ProcessGroup] = None, **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self._process_group = process_group if process_group is not None else dist.group.WORLD

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not dist.is_initialized() or not self.training:
            return super().forward(input)

        return _SyncBatchNormFunction.apply(
            input,
            self.weight,
            self.bias,
            self.affine,
            self.track_running_stats,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self._process_group,
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
