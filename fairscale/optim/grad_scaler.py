# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

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

    def __init__(self) -> None:
        super().__init__()

    def step(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> Optional[float]:
        assert isinstance(optimizer, OSS), "ShardedGradScaler is to be used in combination with a sharded optimizer"

        # Re-use the GradSCaler machinery, but make sure that the status is sync'ed in between the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        handles = [dist.all_reduce(v, async_op=True) for v in optimizer_state["found_inf_per_device"].values()]

        # Make sure that the calls are done before moving out
        _ = list(map(lambda x: x.wait(), handles))

        # Call Torch's GradScaler in turn, states have been synchronized across ranks
        return super().step(optimizer, *args, **kwargs)
