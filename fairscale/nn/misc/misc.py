# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def patch_batchnorm(module: nn.Module) -> None:
    """Patch all batchnorm instances (1d, 2d, 3d, sync_bn, etc.) of a module
       so that they don't track running stats when torch.no_grad() is enabled.

       This is important in activation checkpointing to ensure stats are tracked
       correctly as if there were no activation checkpointing. The reason is
       that activation checkpointing runs the forward function twice, first
       with torch.no_grad(), then with torch.grad().

    Args:
        module (nn.Module):
            The module to be patched in-place.
    """
    # Build a list of BN classes and use two functions only make them track
    # running stats in one of the forward function.
    bn_list = []
    for name, child in module.named_modules():
        # _BatchNorm is base for bn1d, bn2d, bn3d and sync_bn, apex_sync_bn, etc.
        if isinstance(child, _BatchNorm):
            bn_list.append(child)

    def pre_forward() -> None:
        for bn in bn_list:
            bn._track_running_stats_backup = bn.track_running_stats
            bn.track_running_stats = False

    def post_forward() -> None:
        for bn in bn_list:
            bn.track_running_stats = bn._track_running_stats_backup

    # Register the pre/post hooks.
