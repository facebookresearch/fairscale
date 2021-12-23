from enum import Enum
import logging
from typing import List, Tuple

import torch
import torch.nn as nn


def _refresh_per_optimizer_state():
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
        return tuple(scaled_up_grads)


class LayerwiseGradientScaler:
    def __init__(self, growth_factor=2.0 ** 0.001, backoff_factor=0.5 ** 0.5, growth_interval=125):
        self.layer_and_scale_down_factor: List = []
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

        assert self._growth_factor > 1.0, "The growth factor must be > 1.0."
        assert self._backoff_factor < 1.0, "The backoff factor must be < 1.0."

        # self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    # def _register_hooks(self, layer, scale_up_factor: float, scale_down_factor: float) -> None:
    #     helper = GradientHelper(scale_up_factor, scale_down_factor)
    #     layer.register_full_backward_hook(helper.scale_gradients)

    def scale(self, layer_info_list):
        for idx in range(len(layer_info_list) - 1, -1, -1):
            name = layer_info_list[idx].name
            layer = layer_info_list[idx].layer

            scale_down_outputs_multiplier = 1.0 / layer_info_list[idx].scaling_factor
            scale_up_inputs_multiplier = layer_info_list[idx - 1].scaling_factor if idx >= 1 else 1

            self.layer_and_scale_down_factor.append((layer, scale_down_outputs_multiplier))
            helper = GradientHelper(scale_up_inputs_multiplier, scale_down_outputs_multiplier)
            layer_handle = layer.register_full_backward_hook(helper.scale_gradients)

            logging.info(
                "layer = %s, input multiplier = %s, output multiplier = %s "
                % (name, scale_up_inputs_multiplier, scale_down_outputs_multiplier)
            )

    def unscale(self) -> None:
        """
        If there are no infs/nans in the tensors, then unscale the tensors.
        """
        for elt in self.layer_and_scale_down_factor:
            layer, factor = elt[0], elt[1]

            for param in layer.parameters():
                if hasattr(param, "grad"):
                    param.grad = torch.mul(param.grad, factor)

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

    # # this function should update the scaling factor for each layer
    # # clip the scale to the range [2 ** 7, 2 ** 24]
    # def update():
    #     pass
