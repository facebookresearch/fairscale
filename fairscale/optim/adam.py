# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

try:
    from fairscale import fused_adam_cuda  # type: ignore

    class Precision(Enum):
        FULL_PRECISION = auto()
        MIXED_PRECISION = auto()
        MEMORY_EFFICIENT_MIXED_PRECISION = auto()
        PURE_FP16 = auto()

    class Adam(torch.optim.Optimizer):
        state: dict
        defaults: dict
        """
        Implements Adam algorithm. Currently GPU-only.
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
            precision (Precision, optional): One of Precision.FULL_PRECISION,
                Precision.MIXED_PRECISION, Precision.MEMORY_EFFICIENT_MIXED_PRECISION
                or Precision.PURE_FP16. Inferred based on model parameter precision if
                None. (default: None)
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
            precision: Optional[Precision] = None,
        ):
            parameters: List[Any] = list(params)
            self.precision = precision

            if self.precision is None:
                self.precision = (
                    Precision.FULL_PRECISION if parameters[0].dtype == torch.float32 else Precision.MIXED_PRECISION
                )

            if self.precision is not Precision.FULL_PRECISION:
                assert parameters[0].dtype == torch.float16

            self.optim_type = torch.float16 if precision is Precision.PURE_FP16 else torch.float32

            self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore

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
            super().__init__(parameters, defaults)
            self.eps_mode = 0 if eps_inside_sqrt else 1

            self.fp32_param_groups: List[Any] = []
            if self.mixed_precision:
                self._build_fp32_params(parameters)

        def _build_fp32_params(self, params: Any) -> None:
            # create FP32 copy of parameters and grads
            fp32_params = []
            for p in params:
                p32 = torch.nn.Parameter(p.data.float()).to(p.device)
                p32.grad = torch.zeros_like(p32.data)
                fp32_params.append(p32)
            params = fp32_params

            self.fp32_param_groups = []
            param_groups = list(params)
            if not isinstance(param_groups[0], dict):
                param_groups = [{"params": param_groups}]

            for param_group in param_groups:
                params = param_group["params"]
                if isinstance(params, torch.Tensor):
                    param_group["params"] = [params]
                else:
                    param_group["params"] = list(params)

                for name, default in self.defaults.items():
                    param_group.setdefault(name, default)

                params = param_group["params"]

                param_set = set()
                for group in self.param_groups:
                    param_set.update(set(group["params"]))

                self.fp32_param_groups.append(param_group)

        @property
        def supports_memory_efficient_fp16(self) -> bool:
            return True

        @property
        def _step_supports_amp_scaling(self) -> bool:
            return False

        @property
        def mixed_precision(self) -> bool:
            return self.precision is Precision.MIXED_PRECISION

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            super().load_state_dict(state_dict)

            # TODO: Optimizer state gets cast to FP16 and back to FP32 for
            # mixed-precision and memory-efficient mixed-precision. Eventually
            # we want to fix this so that precision is not lost
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["exp_avg"] = self.state[p]["exp_avg"].type(self.optim_type)
                    self.state[p]["exp_avg_sq"] = self.state[p]["exp_avg_sq"].type(self.optim_type)

        def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
            """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
                grads (list of tensors, optional): weight gradient to use for the
                    optimizer update. If gradients have type torch.half, parameters
                    are expected to be in type torch.float. (default: None)
                output params (list of tensors, optional): A reduced precision copy
                    of the updated weights written out in addition to the regular
                    updated weights. Have to be of same type as gradients. (default: None)
                scale (float, optional): factor to divide gradient tensor values
                    by before applying to weights. (default: 1)
            """
            loss = None
            if closure is not None:
                loss = closure()

            for i in range(len(self.param_groups)):
                group = self.param_groups[i]
                bias_correction = 1 if group["bias_correction"] else 0
                tensorlists: Dict[torch.device, List[List[torch.Tensor]]] = dict()

                for j in range(len(group["params"])):
                    p = group["params"][j]
                    # note: p.grad should not ever be set for correct
                    # operation of mixed precision optimizer that sometimes
                    # sends None gradients
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError(
                            "FusedAdam does not support sparse gradients, " "please consider SparseAdam instead"
                        )

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, dtype=self.optim_type)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.optim_type)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1
                    out_p = p.data if self.mixed_precision else torch.tensor([])
                    param = self.fp32_param_groups[i]["params"][j] if self.mixed_precision else p

                    scale = 1.0

                    if self.mixed_precision:
                        pl = [param.data, exp_avg, exp_avg_sq, grad, out_p]
                        if p.device not in tensorlists:
                            tensorlists[p.device] = [[], [], [], [], []]

                        for tl, t in zip(tensorlists[p.device], pl):
                            tl.append(t)
                    else:
                        pl = [param.data, exp_avg, exp_avg_sq, grad]

                        if p.device not in tensorlists:
                            tensorlists[p.device] = [[], [], [], []]

                        for tl, t in zip(tensorlists[p.device], pl):
                            tl.append(t)

                for tensordevice, tensorlist in tensorlists.items():
                    with torch.cuda.device(tensordevice):
                        fused_adam_cuda.adam(
                            2048 * 32,
                            self._overflow_buf,
                            tensorlist,
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


except ImportError:
    pass
