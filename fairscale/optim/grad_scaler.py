from typing import Dict

from torch import Tensor, device
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer


class FairscaleGradScaler(GradScaler):
    def _unscale_grads_(
        self, optimizer: Optimizer, inv_scale: Tensor, found_inf: Tensor, allow_fp16: bool
    ) -> Dict[device, Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16=True)
