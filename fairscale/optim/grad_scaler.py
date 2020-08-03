from typing import Any, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler, _MultiDeviceReplicator


class FairscaleGradScaler(GradScaler):
    def __init__(
        self,
        init_scale: Optional[float] = 2.0 ** 16,
        growth_factor: Optional[float] = 2.0,
        backoff_factor: Optional[float] = 0.5,
        growth_interval: Optional[int] = 2000,
        enabled: Optional[bool] = True,
    ):
        super(FairscaleGradScaler, self).__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.init_scale = init_scale

    def _unscale_grads_(
        self, optimizer: torch.optim.Optimizer, inv_scale: float, found_inf: Any, allow_fp16: bool
    ) -> Any:
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    torch._amp_non_finite_check_and_unscale_(
                        param.grad,
                        per_device_found_inf.get(param.grad.device),
                        per_device_inv_scale.get(param.grad.device),
                    )

        return per_device_found_inf._per_device_tensors
