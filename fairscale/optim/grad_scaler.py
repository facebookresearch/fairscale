# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from torch import Tensor, device
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
from torch.optim import Optimizer

from .oss import OSS


class GradScaler(TorchGradScaler):
    def _unscale_grads_(
        self, optimizer: Optimizer, inv_scale: Tensor, found_inf: Tensor, allow_fp16: bool
    ) -> Dict[device, Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)


class ShardedGradScaler(TorchGradScaler):
    def __init__(self) -> None:
        super().__init__()

    def step(self, optimizer: OSS, *args: Any, **kwargs: Any) -> Optional[float]:
        # Re-use the GradSCaler machinery, but make sure that the status is sync'ed in between the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]  # type: ignore

        for v in optimizer_state["found_inf_per_device"].values():
            dist.all_reduce(v)

        return super().step(optimizer, *args, **kwargs)  # type: ignore
