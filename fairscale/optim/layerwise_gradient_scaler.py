import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class LayerInfo:
    def __init__(
        self, name: str, layer: nn.Module, scale: float = 1.0) -> None:
        self.name = name
        self.layer = layer
        self.scale_up = scale
        self.found_inf_or_nan = False


class GradientHelper:
    def __init__(self, name:str, inputs_multiplier: float, outputs_multiplier: float):
        self.name = name
        self.inputs_multiplier = inputs_multiplier
        self.outputs_multiplier = outputs_multiplier

    def scale_gradients(self, m: nn.Module, inputs: Tuple, outputs: Tuple) -> Tuple[torch.Tensor]:
        # scale up the inputs
        scaled_up_grads = list()
        for idx in range(len(inputs)):
            if inputs[idx] is not None:
                if self.inputs_multiplier != 1.0 or self.outputs_multiplier != 1.0:
                    logging.debug("layer = %s \t scale_up = %s \t scale_down = %s" % (self.name, self.inputs_multiplier, self.outputs_multiplier))
                scaled_up_grads.append(inputs[idx].mul(self.inputs_multiplier * self.outputs_multiplier))
            else:
                logging.debug("next layer is None")
                scaled_up_grads.append(inputs[idx])
        return tuple(scaled_up_grads)  # type: ignore


class LayerwiseGradientScaler:
    def __init__(  # type: ignore
        self,
        model,
        layer_scale_dict: dict,
        growth_factor: float = 4.0,
        backoff_factor: float = 0.5 ** 0.5
    ) -> None:
        self._model = model
        self._layer_scale_dict: dict = layer_scale_dict
        self._growth_factor: float = growth_factor
        self._backoff_factor: float = backoff_factor
        self._apply_layerwise_scaling: bool = True if len(layer_scale_dict.keys()) > 0 else False
        self._handles: List = []
        self.layer_info: List = []

        if self._apply_layerwise_scaling:
            assert self._growth_factor > 1.0, "The growth factor must be > 1.0."
            assert self._backoff_factor < 1.0, "The backoff factor must be < 1.0."
            self.layer_info = self._build_layer_info()

    def _build_layer_info(self) -> List:
        layer_info_list = list()

        for name, layer in self._model.named_modules():
            if name != "":
                if name not in self._layer_scale_dict.keys():
                    logging.debug("name = %s, layer = %s, scaling_factor = %s" % (name, layer, 1.0))
                    layer_info_list.append(LayerInfo(name, layer, 1.0))
                else:
                    logging.debug(
                        "name = %s, layer = %s, scaling_factor = %s" % (name, layer, self._layer_scale_dict[name])
                    )
                    layer_info_list.append(LayerInfo(name, layer, self._layer_scale_dict[name]))
        return layer_info_list

    def scale(self) -> None:
        if self._apply_layerwise_scaling:
            for idx in range(len(self.layer_info)):
                elt = self.layer_info[idx]
                name, layer = elt.name, elt.layer
                
                inputs_multiplier = 1.0
                if idx > 0:
                    inputs_multiplier = self.layer_info[idx-1].scale_up
                
                outputs_multiplier = 1.0 / elt.scale_up
                helper = GradientHelper(name, inputs_multiplier, outputs_multiplier)
                layer_handle = layer.register_full_backward_hook(helper.scale_gradients)
                self._handles.append(layer_handle)
                logging.debug("name = %s \t scale = %s" % (name, elt.scale_up))

       # for name, layer in self._model.named_modules():
       #     if name != "":
       #         print(name, layer._get_backward_hooks())

    def unscale(self) -> None:
        """
        If there are no infs/nans in the tensors, then unscale the tensors.
        """
        if self._apply_layerwise_scaling:
            for elt in self.layer_info:
                for param_name, param in elt.layer.named_parameters():
                    if hasattr(param, "grad"):
                        if torch.isinf(param.grad).any().item():
                            logging.debug('found inf, skipping unscale')
                        else:
                            logging.debug("%s scaling down %s by %s" % (elt.name, param_name, 1.0 / elt.scale_up))
                            param.grad = torch.mul(param.grad, 1.0 / elt.scale_up)

            while len(self._handles) > 0:
                elt = self._handles.pop()
                elt.remove()

    def _check_for_inf_or_nan(self) -> None:  # type: ignore
        """
        Check for infs/nans in each tensor of the gradient. 
        If a tensor contains inf/nan then layer.found_inf_or_nan is set to True
        """
        for elt in self.layer_info:
            elt.found_inf_or_nan = False
            for _, param in elt.layer.named_parameters():
                if hasattr(param, "grad"):
                    for tensor in param.grad:
                        if torch.isinf(tensor).any().item() is True or torch.isnan(tensor).any().item() is True:  # type: ignore
                            elt.found_inf_or_nan = True
                            break # skip all remaining tensors for this named parameter
                if elt.found_inf_or_nan is True:
                    break # skip all remaining named parameter of this layer

    def step(self, optimizer, count) -> None:  # type: ignore
        """
        If there are NO infs/nans in the gradient of all layers, then optimizer takes a step.
        Update the scaling factors for each layer.
        """
        if self._apply_layerwise_scaling:
            self._check_for_inf_or_nan()
            inf_nan_found = any(elt.found_inf_or_nan for elt in self.layer_info)
        
            if not inf_nan_found:
                optimizer.step()
            else:
                logging.info("inf found at step %s" % count)
            self._update_scale()
        else:
            optimizer.step()

    def _update_scale(self) -> None:
        """
        Update the scaling factor for each layer and clip the scale to the range [2 ** 7, 2 ** 24]
        """
        if self._apply_layerwise_scaling:
            for elt in self.layer_info:
                if not elt.found_inf_or_nan:
                    if elt.scale_up != 1.0:
                        elt.scale_up = max(2 ** 7, min(self._growth_factor * elt.scale_up, 2 ** 24))
                else:
                    if elt.scale_up != 1.0:
                        elt.scale_up = max(2 ** 7, min(self._backoff_factor * elt.scale_up, 2 ** 24))

