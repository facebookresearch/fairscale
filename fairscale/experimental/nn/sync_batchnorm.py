# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fairscale.nn.checkpoint import is_checkpointing, is_recomputing
from fairscale.utils import torch_version


def _forward(input: Tensor, affine: bool, mean: Tensor, invstd: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    if affine:
        return (input - mean) * (invstd * weight.reshape_as(mean)) + bias.reshape_as(mean)
    else:
        return (input - mean) * invstd


def _track_running_stats(
    running_mean: Tensor, running_var: Tensor, momentum: float, mean: Tensor, var: Tensor, total_count: Tensor
) -> None:
    unbiased_var = var * (total_count / (total_count - 1))
    running_mean += momentum * (mean.reshape(-1) - running_mean)
    running_var += momentum * (unbiased_var.reshape(-1) - running_var)


def _calculate_stats(input: Tensor, eps: float, process_group: ProcessGroup) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    return mean, var, invstd, total_count


if torch_version()[:2] >= (1, 7):
    _forward = torch.jit.script(_forward)  # type: ignore
    _track_running_stats = torch.jit.script(_track_running_stats)  # type: ignore


class _SyncBatchNormFunction(torch.autograd.Function):
    """
    An autograd function used to avoid storing activations for intermediate results.

    NOTE: Even though the mean and var are passed into this function, we do the entire
    backward, including mean and var, here. We have to calculate statistics outside
    this function in order to avoid multiple all_reduces when using checkpointing.
    """

    @staticmethod
    # type: ignore
    def forward(ctx, input, weight, bias, affine, mean, invstd, total_count, process_group):
        ctx.save_for_backward(input, weight, bias, mean, invstd, total_count)
        ctx.process_group = process_group

        return _forward(input, affine, mean, invstd, weight, bias)

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
        self.saved_for_2nd_fwd: List[Tuple] = []
        self.disable_patch_batchnorm = True

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        # There are 3 modes this is being called:
        # 1. not wrapped (and there is only a single phase)
        # 2. wrapped and in checkpointing phase
        # 3. wrapped and in recomputing phase

        if not dist.is_initialized() or not self.training:
            return super().forward(input)

        wrapped = is_checkpointing() or is_recomputing()
        if not wrapped or is_checkpointing():
            # NOTE The full backward, including mean and var, is done by _SyncBatchNormFunction.
            with torch.no_grad():
                mean, var, invstd, total_count = _calculate_stats(input, self.eps, self._process_group)
                if self.track_running_stats:
                    _track_running_stats(self.running_mean, self.running_var, self.momentum, mean, var, total_count)

        if is_checkpointing():
            self.saved_for_2nd_fwd.append((mean, invstd, total_count))
            return _forward(input, self.affine, mean, invstd, self.weight, self.bias)
        if is_recomputing():
            mean, invstd, total_count = self.saved_for_2nd_fwd.pop(0)

        return _SyncBatchNormFunction.apply(
            input, self.weight, self.bias, self.affine, mean, invstd, total_count, self._process_group
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
