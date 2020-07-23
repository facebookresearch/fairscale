from typing import TYPE_CHECKING, Any, Callable, List, Optional

from apex.multi_tensor_apply import multi_tensor_applier
from apex.optimizers import FusedAdam
import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class FusedAdamV2(FusedAdam):
    state: dict

    """
    Compared to the original version in Apex, the fairseq version casts grads
    and params to FP32 internally to support ``--memory-efficient-fp16``.
    """

    def __init__(self, *args: Optional[List], **kwargs: Optional[dict]):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "multi_tensor_adam"):
            raise Exception("Apex installation is outdated. Please install an updated version of apex.")

        property

    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None, scale: Optional[float] = 1.0,) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            # create lists for multi-tensor apply
            g_16, p_16, orig_p_16, m_16, v_16 = [], [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, " "please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float)
                else:
                    state["exp_avg"] = state["exp_avg"].to(device=p.data.device, dtype=torch.float)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(device=p.data.device, dtype=torch.float)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data.float())
                    p_16.append(p.data.float())
                    orig_p_16.append(p.data)
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state["exp_avg"])
                    v_32.append(state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedAdam only support fp16 and fp32.")

            with torch.cuda.device(p.device):
                if len(g_16) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )
                    for orig_p, p in zip(orig_p_16, p_16):
                        orig_p.copy_(p.data)
                if len(g_32) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss
