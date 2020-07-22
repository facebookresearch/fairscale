# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

fused_adam_cuda: Any


class FusedAdamV1(torch.optim.Optimizer):
    state: dict

    """
    Implements Adam algorithm. Currently GPU-only. Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Compared to the original version in Apex, the fairseq version casts grads
    and params to FP32 internally to support ``--memory-efficient-fp16``.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params: _params_t,
        lr: Optional[float] = 1e-3,
        bias_correction: Optional[bool] = True,
        betas: Optional[Tuple[float, float]] = (0.9, 0.999),
        eps: Optional[float] = 1e-8,
        eps_inside_sqrt: Optional[bool] = False,
        weight_decay: Optional[float] = 0.0,
        max_grad_norm: Optional[float] = 0.0,
        amsgrad: Optional[bool] = False,
    ):
        global fused_adam_cuda
        import importlib

        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
        }
        super().__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return True

    @property
    def supports_step_with_scale(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None, scale: Optional[float] = 1.0,) -> Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group.get("bias_correction", 1) else 0

            for p in group["params"]:
                # note: p.grad should not ever be set for correct
                # operation of mixed precision optimizer that sometimes
                # sends None gradients
                if p.grad is None:
                    continue
                grad = p.grad.data

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = p.data
                with torch.cuda.device(p.device):
                    fused_adam_cuda.adam(
                        p_data_fp32,
                        out_p,
                        exp_avg,
                        exp_avg_sq,
                        grad,
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        scale,
                        state["step"],
                        self.eps_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss
