# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
To prevent underflow or overflow of gradients, DynamicLossScaler is used to
dynamically scale up and down gradients by scaling the loss. The usage of the
DynamicLossScaler is similar with the GradScaler except that DynamicLossScaler
can be used for updates on a CPU device.
https://pytorch.org/docs/stable/_modules/torch/cuda/amp/grad_scaler.html#GradScaler
"""


from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional

import torch


class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_per_optimizer_state() -> OptState:
    return OptState.READY


class DynamicLossScaler(object):
    """An instance ``scaler`` helps perform the steps of gradient scaling
    conveniently.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        tolerance: float = 0.0,
        threshold: float = None,
        min_loss_scale: float = 1e-4,
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold
        self.min_loss_scale = min_loss_scale
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0
        self._per_optimizer_states: Dict[int, OptState] = defaultdict(_refresh_per_optimizer_state)
        self._scale = None

    def scale(self, outputs):  # type: ignore
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.

        Returns:
            Tensor or iterable of Tensors: Scaled outputs.
        """
        return self.loss_scale * outputs

    @torch.no_grad()
    def _get_gradients_norm(self, params: List[torch.nn.Parameter]) -> float:
        grads = []
        for p in params:
            if p.grad is None:
                continue
            else:
                grads.append(p.grad.detach())

        if len(grads) == 0:
            return 0.0

        if len(grads) == 1:
            total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)  # type: ignore
        else:
            total_norm = torch.norm(torch.stack([torch.norm(g, p=2, dtype=torch.float32) for g in grads]))  # type: ignore
        return total_norm.item()

    def _decrease_loss_scale(self) -> None:
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    def _check_overflow(self, grad_norm: float) -> None:
        # detect inf and nan
        if grad_norm == float("inf") or grad_norm != grad_norm:
            # overflow has occured
            prev_scale = self.loss_scale
            iter_since_rescale = self._iter - self._last_rescale_iter

            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0

            if self.loss_scale <= self.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                self.loss_scale = prev_scale
                raise FloatingPointError(
                    (
                        "Minimum loss scale reached ({}). Your loss is probably exploding. "
                        "Try lowering the learning rate, using gradient clipping or "
                        "increasing the batch size."
                    ).format(self.min_loss_scale)
                )

            self._iter += 1
            raise OverflowError("setting loss scale to: " + str(self.loss_scale))

    def update(self) -> None:
        """Updates the scale factor."""

        if (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def step(self, optimizer, *args, **kwargs):  # type: ignore
        """
        :meth:`step` unscale the gradients and step the optimizer.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        Returns:
            The return value of ``optimizer.step(*args, **kwargs)``.  None when overflow or underflow
            gradients occur and optimizer.step() is skipped.
        """
        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if DynamicLossScaler is enabled.")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        # check gradient norm. If gradient norm is nan or inf, adjust scale here, and skip step.
        # clip_grads_norm can happen before this step
        for group in optimizer.param_groups:
            grad_norm = self._get_gradients_norm(group["params"])
            try:
                self._check_overflow(grad_norm)
            except OverflowError:
                return None

        if optimizer_state is OptState.READY:
            self.unscale_(optimizer)
            state_dict = optimizer.state_dict()
            state_dict["loss_scale"] = self.loss_scale

        retval = optimizer.step(*args, **kwargs)

        optimizer_state = OptState.STEPPED

        return retval

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        # uncale the gradients.
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        elif optimizer_state is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        assert self.loss_scale is not None
        inv_scale = 1.0 / float(self.loss_scale)

        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    else:
                        param.grad.data.mul_(inv_scale)

        optimizer_state = OptState.UNSCALED

    def state_dict(self) -> Optional[Dict[str, float]]:
        if self.loss_scale is not None:
            return {"loss_scale": self.loss_scale}

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        if "loss_scale" in state_dict:
            self.loss_scale = state_dict["loss_scale"]
