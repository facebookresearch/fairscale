import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class LayerInfo:
    def __init__(
        self, name: str, layer: nn.Module, scale_up: float = 1.0, scale_down: float = 1.0, growth_tracker: int = 0
    ) -> None:
        self.name = name
        self.layer = layer
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.growth_tracker = growth_tracker


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
    def __init__(  # type: ignore
        self,
        model,
        layer_scale_dict: dict,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 125,
    ) -> None:
        self._model = model
        self._layer_scale_dict = layer_scale_dict
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._num_layers_to_scale = len(layer_scale_dict.keys())
        self.handles: List = []

        if self._num_layers_to_scale > 0:
            assert self._growth_factor > 1.0, "The growth factor must be > 1.0."
            assert self._backoff_factor < 1.0, "The backoff factor must be < 1.0."
            self.layer_info = self._build_layer_info()

    def _build_layer_info(self) -> List:
        layer_info_list = list()

        for name, layer in self._model.named_modules():
            if name != "":
                if name not in self._layer_scale_dict.keys():
                    logging.info("name = %s, layer = %s, scaling_factor = %s" % (name, layer, 1.0))
                    layer_info_list.append(LayerInfo(name, layer, 1.0))
                else:
                    logging.info(
                        "name = %s, layer = %s, scaling_factor = %s" % (name, layer, self._layer_scale_dict[name])
                    )
                    layer_info_list.append(LayerInfo(name, layer, self._layer_scale_dict[name]))
        return layer_info_list

    def scale(self) -> None:
        if self._num_layers_to_scale > 0:
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
                if elt.scale_up != 1.0:
                    logging.debug(
                        "layer name = %s, scale_up = %s, scale_down = %s, input multiplier = %s, output multiplier = %s "
                        % (
                            name,
                            elt.scale_up,
                            elt.scale_down,
                            scale_up_inputs_multiplier,
                            scale_down_outputs_multiplier,
                        )
                    )

    def unscale(self) -> None:
        """
        If there are no infs/nans in the tensors, then unscale the tensors.
        """
        if self._num_layers_to_scale > 0:
            for elt in reversed(self.layer_info):
                for param_name, param in elt.layer.named_parameters():
                    if hasattr(param, "grad"):
                        logging.debug("%s scaling down %s by %s" % (elt.name, param_name, elt.scale_down))
                        param.grad = torch.mul(param.grad, elt.scale_down)

            while len(self.handles) > 0:
                elt = self.handles.pop()
                elt.remove()

    def _check_for_inf_or_nan(self, model) -> Dict:  # type: ignore
        """
        Check for infs/nans in each tensor of the gradient. Return True if infs/nans exist else False
        save layer name and if this layer's gradient has an inf/nan value.
        """
        module_name_found_inf = dict()
        for module_name, layer in model.named_modules():
            if module_name != "":
                module_name_found_inf[module_name] = False
                for _, param in layer.named_parameters():
                    if hasattr(param, "grad"):
                        for tensor in param.grad:
                            if torch.isinf(tensor).any().item() is True or torch.isnan(tensor).any().item() is True:  # type: ignore
                                module_name_found_inf[module_name] = True
                                break
                    if module_name_found_inf[module_name] is True:
                        break
        return module_name_found_inf

        # per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        # with torch.no_grad():
        #    # layout gradients by device and dtype
        #    for group in optimizer.param_groups:
        #        for param in group["params"]:
        #            if param.grad is None:
        #                continue
        #            else:
        #                per_device_and_dtype_grads[param.grad.device][param.grad.dtype].append(param.grad)

        #    # check for each tensor in the gradients
        #    # if inf/nan is found return True else False
        #    for device, per_dtype_grads in per_device_and_dtype_grads.items():
        #        for grads in per_dtype_grads.values():
        #            for tensor in grads:
        #                if torch.isinf(tensor).any().item() is True or torch.isnan(tensor).any().item() is True:  # type: ignore
        #                    return True
        #    return False

    def step(self, model, optimizer) -> None:  # type: ignore
        """
        If there are no infs/nans in the gradient, then optimizer takes a step.
        Update the scaling factors for each layer depending on the presence of
        infs/nans.
        """
        module_name_found_inf_nan_dict = self._check_for_inf_or_nan(model)
        inf_nan_found = any(module_name_found_inf_nan_dict.values())
        logging.debug(module_name_found_inf_nan_dict)
        if not inf_nan_found:
            optimizer.step()
        else:
            logging.debug("inf found")
        self._update_scale(module_name_found_inf_nan_dict)

    def _update_scale(self, module_name_found_inf_nan_dict: dict) -> None:
        """
        Update the scaling factor for each layer and clip the scale to the range [2 ** 7, 2 ** 24]
        """
        if self._num_layers_to_scale > 0:
            logging.debug("updating scale")
            for elt in self.layer_info:
                inf_nan_found_in_layer = module_name_found_inf_nan_dict[elt.name]
                if not inf_nan_found_in_layer:
                    if elt.scale_up != 1.0:
                        elt.scale_up = min(self._growth_factor * elt.scale_up, 2 ** 24)
                    if elt.scale_down != 1.0:
                        elt.scale_down = max(1.0 / self._growth_factor, 2 ** 7)
                else:
                    if elt.scale_up != 1.0:
                        elt.scale_up = min(self._backoff_factor * elt.scale_up, 2 ** 24)
                    if elt.scale_down != 1.0:
                        elt.scale_down = max(1.0 / self._backoff_factor, 2 ** 7)
