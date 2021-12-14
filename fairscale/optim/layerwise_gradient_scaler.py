import logging
from typing import List, Tuple

import torch
import torch.nn as nn


class GradientHelper:
    def __init__(self, scale_up_factor: float, scale_down_factor: float):
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor

    def scale_gradients(self, m: nn.Module, inputs: Tuple, outputs: Tuple) -> Tuple[torch.Tensor]:
        # scale up the inputs
        scaled_up_grads = list()
        for idx in range(len(inputs)):
            if inputs[idx] is not None:
                scaled_up_grads.append(inputs[idx].mul(self.scale_up_factor * self.scale_down_factor))
            else:
                logging.debug("inputs[%d] is None" % idx)
                scaled_up_grads.append(inputs[idx])
        return tuple(scaled_up_grads)


class LayerwiseGradientScaler:
    def __init__(self) -> None:
        self.layer_and_scale_down_factor: List = []

    def register_hooks(self, layer, scale_up_factor: float, scale_down_factor: float) -> None:
        helper = GradientHelper(scale_up_factor, scale_down_factor)
        layer.register_full_backward_hook(helper.scale_gradients)

    def scale(self, layers_to_scale: List, scaling_factors: List) -> None:
        for idx in range(len(scaling_factors) - 1, -1, -1):
            scale_up_inputs_multiplier = scaling_factors[idx - 1] if idx >= 1 else 1
            scale_down_outputs_multiplier = 1.0 / scaling_factors[idx]

            # name and layer tuple
            _, layer = layers_to_scale[idx][0], layers_to_scale[idx][1]

            self.layer_and_scale_down_factor.append((layer, scale_down_outputs_multiplier))
            self.register_hooks(layer, scale_up_inputs_multiplier, scale_down_outputs_multiplier)

            logging.info(
                "layer = %s, input multiplier = %s, output multiplier = %s "
                % (layers_to_scale[idx][0], scale_up_inputs_multiplier, scale_down_outputs_multiplier)
            )

    def unscale(self) -> None:
        # scale down the outputs
        for elt in self.layer_and_scale_down_factor:
            layer, factor = elt[0], elt[1]

            for param in layer.parameters():
                if hasattr(param, "grad"):
                    param.grad = torch.mul(param.grad, factor)
