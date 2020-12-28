# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
from torch.optim import Optimizer

from .oss import OSS


class GradScaler(TorchGradScaler):
    def _unscale_grads_(
        self, optimizer: Optimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool
    ) -> Dict[torch.device, torch.Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)


class ShardedGradScaler(TorchGradScaler):
    """
    A shard-aware :class:`GradScaler<torch.cuda.amp.GradScaler>`, to be used in conjunction with
    :class:`OSS` and :class:`ShardedOptimizer`.

    Interface and usecases are not changed, more explanations can be found in the corresponding pytorch
    documentation https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
    """

    def __init__(self, process_group: Any = dist.group.WORLD) -> None:
        super().__init__()
        self.display_warning = True
        self.group = process_group

    def unscale_(self, optimizer: Optimizer) -> None:
        # Could be a mistake, this scaler is supposed to work with ZeroRedundancyOptimizer only
        if self.display_warning and not isinstance(optimizer, OSS):
            logging.warning(
                "ShardedGradScaler is to be used in combination with a sharded optimizer, this could not be checked"
            )

        self.display_warning = False  # Only warn once

        # Call the upstream unscale_ method which will only act on this rank's gradients
        super().unscale_(optimizer)

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        handles = [
            dist.all_reduce(v, async_op=True, group=self.group)
            for v in optimizer_state["found_inf_per_device"].values()
        ]

        # Make sure that the calls are done before moving out
        _ = list(map(lambda x: x.wait(), handles))
