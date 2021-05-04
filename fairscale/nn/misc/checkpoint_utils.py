# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


def patch_batchnorm(module: nn.Module) -> List:
    """Patch all batchnorm instances (1d, 2d, 3d, sync_bn, etc.) of a module
       so that they don't track running stats when torch.no_grad() is enabled.

       This is important in activation checkpointing to ensure stats are tracked
       correctly as if there were no activation checkpointing. The reason is
       that activation checkpointing runs the forward function twice, first
       with torch.no_grad(), then with torch.grad().

    Args:
        module (nn.Module):
            The module to be patched in-place.

    Returns:
        (list):
            A list of hook handles, late can be freed.
    """

    def pre_forward(module: _BatchNorm, input: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module._track_running_stats_backup = module.track_running_stats
        module.track_running_stats = False

    def post_forward(module: _BatchNorm, input: Tensor, result: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module.track_running_stats = module._track_running_stats_backup

    hooks = []
    for name, child in module.named_modules():
        # _BatchNorm is base for bn1d, bn2d, bn3d and sync_bn, apex_sync_bn, etc.
        if isinstance(child, _BatchNorm):
            # Register the pre/post hooks.
            pre_handle = child.register_forward_pre_hook(pre_forward)
            post_handle = child.register_forward_hook(post_forward)
            hooks += [pre_handle, post_handle]
    return hooks


def init_counter(module: nn.Module) -> None:
    """Add a checkpoint forward pass counter to a module and all its child FSDP modules.

       ``inc_counter`` and ``dec_counter`` are used together with this to maintain counters
       for FSDP to use in case of multiple forward pass and checkpoint being used at the same time.
    """
    for mod in module.modules():
        mod._checkpoint_fwd_counter = 0


def _add_counter(module: nn.Module, value: int) -> None:
    if not hasattr(module, "_checkpoint_fwd_counter"):
        return
    for mod in module.modules():
        mod._checkpoint_fwd_counter += value


def inc_counter(module: nn.Module) -> None:
    _add_counter(module, 1)


def dec_counter(module: nn.Module) -> None:
    _add_counter(module, -1)
