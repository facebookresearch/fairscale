from enum import Enum
import logging
from typing import List, Tuple

import torch
import torch.nn as nn


def _refresh_per_optimizer_state() -> dict:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2


class GradientHelper:
    def __init__(self, scale_up_factor: float, scale_down_factor: float):
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor

    def scale_gradients(self, m: nn.Module, inputs: Tuple, outputs: Tuple) -> Tuple[torch.Tensor]:
        # scale up the inputs
        scaled_up_grads = list()
        for idx in range(len(inputs)):
            if inputs[idx] is not None:
                logging.debug("scale up factor is: %s" % self.scale_up_factor)
                logging.debug("scale down factor is: %s" % self.scale_down_factor)
                scaled_up_grads.append(inputs[idx].mul(self.scale_up_factor * self.scale_down_factor))
            else:
                logging.debug("inputs[%d] is None" % idx)
                scaled_up_grads.append(inputs[idx])
        return tuple(scaled_up_grads)  # type: ignore


class LayerwiseGradientScaler:
    def __init__(
        self,
        layer_info: List,
        growth_factor: float = 2.0 ** 0.001,
        backoff_factor: float = 0.5 ** 0.5,
        growth_interval: int = 125,
    ) -> None:
        self.layer_info = layer_info
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self.handles: List = []

        assert self._growth_factor > 1.0, "The growth factor must be > 1.0."
        assert self._backoff_factor < 1.0, "The backoff factor must be < 1.0."
        # self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def scale(self) -> None:
        for idx in range(len(self.layer_info) - 1, -1, -1):
            elt = self.layer_info[idx]
            name = elt.name
            layer = elt.layer

            scale_down_outputs_multiplier = 1.0 / elt.scale_up
            scale_up_inputs_multiplier = self.layer_info[idx - 1].scale_up if idx >= 1 else 1
            elt.scale_down = scale_down_outputs_multiplier

            helper = GradientHelper(scale_up_inputs_multiplier, scale_down_outputs_multiplier)
            layer_handle = layer.register_full_backward_hook(helper.scale_gradients)
            self.handles.append(layer_handle)

            logging.info(
                "layer name = %s, scale_up = %s, scale_down = %s, input multiplier = %s, output multiplier = %s "
                % (name, elt.scale_up, elt.scale_down, scale_up_inputs_multiplier, scale_down_outputs_multiplier)
            )

    def unscale(self) -> None:
        """
        If there are no infs/nans in the tensors, then unscale the tensors.
        """
        for elt in reversed(self.layer_info):
            for param_name, param in elt.layer.named_parameters():
                if hasattr(param, "grad"):
                    logging.debug("%s scaling down %s by %s" % (elt.name, param_name, elt.scale_down))
                    param.grad = torch.mul(param.grad, elt.scale_down)

        while len(self.handles) > 0:
            elt = self.handles.pop()
            elt.remove()

    def update(self) -> None:
        # this function should update the scaling factor for each layer
        # clip the scale to the range [2 ** 7, 2 ** 24]
        # ensure that scale value is same on all gpus inside the update function

        for elt in self.layer_info:
            if elt.scale_up != 1.0:
                elt.scale_up = min(2 * elt.scale_up, 2 ** 24)
            if elt.scale_down != 1.0:
                elt.scale_down = max(0.5 * elt.scale_down, 2 ** 7)

    # def _unscale_grads_(self):
    #     pass

    # def unscale_(self, optimizer):
    #     """
    #     checks for infs/nans in the gradient tensors
    #     """
    #     optimizer_state = self._per_optimizer_states[id(optimizer)]

    #     if optimizer_state["stage"] is OptState.UNSCALED:
    #         raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
    #     elif optimizer_state["stage"] is OptState.STEPPED:
    #         raise RuntimeError("unscale_() is being called after step().")

    # def _maybe_optimizer_step(self, optimizer, optimizer_state):
    #     retval = None
    #     if not sum(v.items for v in optimizer_state["found_inf_per_device"].value()):
    #         retval = optimizer.step()
    #     return retval

    # # unscale the gradients, if gradients don't contain inf/nan then optimizer.step is called,
    # # otherwise optimizer.step is skipped
    # def step(self, optimizer):
    #     """
    #     1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
    #         earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
    #     2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
    #         gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.
    #     """

    #     # self._check_scale_growth_tracker("step")
    #     optimizer_state = self._per_optimizer_states[id(optimizer)]

    #     if optimizer_state["stage"] is OptState.STEPPED:
    #         raise RuntimeError("step() has already been called since the last update().")

    #     if optimizer_state["stage"] is OptState.READY:
    #         self.unscale_(optimizer)

    #     assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

    #     retval = self._maybe_optimizer_step(optimizer, optimizer_state)
    #     optimizer_state["stage"] = OptState.STEPPED
    #     return retval
