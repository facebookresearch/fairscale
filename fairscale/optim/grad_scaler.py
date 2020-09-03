# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torch import Tensor, device
from torch.cuda.amp import GradScaler as TorchGradScaler
from torch.optim import Optimizer


class GradScaler(TorchGradScaler):
    def _unscale_grads_(
        self, optimizer: Optimizer, inv_scale: Tensor, found_inf: Tensor, allow_fp16: bool
    ) -> Dict[device, Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)
