import logging
from typing import List, Tuple

import torch
import torch.nn as nn


class LayerInfo:
    """
    A class to record the layer attributes.
    """

    def __init__(self, name: str, layer: nn.Module, scale: float = 1.0, scale_layer: bool = False) -> None:
        """
        layer_name: name of the layer e.g. fc1, conv1, relu1
        layer: type of the layer e.g. Linear, Conv2d, ReLU
        scaling_factor: user configurable scaling factor for the layer, defaults to 1.0
        found_inf_or_nan: a boolean indicating if any parameter of layer's gradient contains inf/nan
        growth_tracker: tracks number of step since last time scale was increased
        scale_layer: a boolean indicating if the layer should be scaled or not
        """
        self.layer_name = name
        self.layer = layer
        self.scaling_factor = scale
        self.found_inf_or_nan = False
        self.growth_tracker = 0
        self.scale_layer = scale_layer


class GradientHelper:
    """
    A helper class to create instances of backward hooks. The hooks are registered in the
    scale method of LayerwiseGradientScaler.
    """

    def __init__(self, name: str, inputs_multiplier: float, outputs_multiplier: float):
        self.layer_name = name
        self.inputs_multiplier = inputs_multiplier
        self.outputs_multiplier = outputs_multiplier

    def scale_gradients(self, m: nn.Module, inputs: Tuple, outputs: Tuple) -> Tuple[torch.Tensor]:
        """
        Backward hook that is attached to the layers to scale the gradients.
        """
        scaled_up_grads = list()
        for idx in range(len(inputs)):
            if inputs[idx] is not None:
                if self.inputs_multiplier != 1.0 or self.outputs_multiplier != 1.0:
                    logging.debug(
                        "layer = %s \t scale = %s \t scale_down = %s"
                        % (self.layer_name, self.inputs_multiplier, self.outputs_multiplier)
                    )
                scaled_up_grads.append(inputs[idx].mul(self.inputs_multiplier * self.outputs_multiplier))
            else:
                logging.debug("next layer is None")
                scaled_up_grads.append(inputs[idx])
        return tuple(scaled_up_grads)  # type: ignore


class LayerwiseGradientScaler:
    """
    LayerwiseGradientScaler enables using distinct scaling factors for each layer
    of the network.

    Example:

    # Create a convolutional network
        class ConvNet(nn.Module):
            def __init__(self):
                ...

            def forward(self, x):
                ...

        # Create an instance of the model
        model = ConvNet()
        optimizer = torch.optim.SGD(model.parameters())

        # specify the layers to scale and their scaling factor
        layer_scale_dict = {"conv1": 2**10, "conv2": 2**8, "fc1": 2**10, "fc2": 2**9}
        scaler = LayerwiseGradientScaler(model, layer_scale_dict)

        for epoch in num_epochs:
            for inputs, targets in batch:
                optimizer.zero_grad()

                # scale the gradients
                scaler.scale()

                # enables mixed precision training
                with autocast():
                    predictions = model(inputs)
                    loss = loss_function(predictions, targets)

                loss.backward()

                # unscale the gradients
                loss.unscale()

                # step is taken if there are no inf/nan in the gradients
                # scaling factor for each layer are updated
                loss.step(optimizer)

    Args:
        model                      : instance of a Model class, such as ConvNet above
        layer_scale_dict (dict)    : dictionary with key = layer_name and value = scaling_factor
        growth_factor (float)      : per layer scaling factor multiplier
        backoff_factor (float)     : per layer scaling factor multiplier when an inf/nan is found
        growth_interval (int)      : number of steps after which scale is multiplied by growth_factor
        min_scaling_factor (float) : smallest scaling factor
        max_scaling_factor (float) : largest scaling factor
    """

    def __init__(  # type: ignore
        self,
        model,
        layer_scale_dict: dict,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 10000,
        min_scale: float = torch.finfo(torch.float32).tiny,  # type: ignore
        max_scale: float = torch.finfo(torch.float32).max,  # type: ignore
    ) -> None:
        self._model = model
        self._layer_scale_dict: dict = layer_scale_dict
        self._growth_factor: float = growth_factor
        self._backoff_factor: float = backoff_factor
        self._growth_interval: int = growth_interval
        self._apply_layerwise_scaling: bool = True if len(layer_scale_dict.keys()) > 0 else False
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._handles: List = []
        self.layer_info: List = []

        if self._apply_layerwise_scaling:
            assert self._growth_factor > 1.0, "The growth factor must be > 1.0."
            assert self._backoff_factor < 1.0, "The backoff factor must be < 1.0."
            self.layer_info = self._build_layer_info()

    def _build_layer_info(self) -> List:
        """
        Helper function to create a list of LayerInfo instances.
        """
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
                    layer_info_list.append(LayerInfo(name, layer, self._layer_scale_dict[name], True))
        return layer_info_list

    def scale(self) -> None:
        """
        For each layer calculates the scaling factor for preceding layer's grad inputs
        and current layer's grad outputs. These values are used to register a full backward
        hook. The handle returned from registering the backward hook is appended to a list
        of handles. New hooks are created and registered at every step and a new list of
        handles is created. The handles are flushed out in the unscale function.
        """
        if not self._apply_layerwise_scaling:
            return

        for idx in range(len(self.layer_info)):
            elt = self.layer_info[idx]
            layer_name, layer = elt.layer_name, elt.layer

            inputs_multiplier = 1.0
            if idx > 0:
                inputs_multiplier = self.layer_info[idx - 1].scaling_factor

            outputs_multiplier = 1.0 / elt.scaling_factor
            helper = GradientHelper(layer_name, inputs_multiplier, outputs_multiplier)
            layer_handle = layer.register_full_backward_hook(helper.scale_gradients)
            self._handles.append(layer_handle)
            logging.debug("name = %s \t scale = %s" % (layer_name, elt.scaling_factor))

    def _get_layers_with_finite_values(self) -> List[LayerInfo]:
        layers_with_finite_values: List = []
        for item in self.layer_info:
            if not item.found_inf_or_nan:
                layers_with_finite_values.append(item)
        return layers_with_finite_values

    def unscale(self) -> None:
        """
        For each layer, check if any of the layer's parameters contain an inf/nan.
        If there are no inf/nan in the gradient, then gradient of that layer is
        unscaled by the reciprocal of the scaling factor for that layer.
        Finally, all handles recorded while registering the hooks are deleted.
        """
        if not self._apply_layerwise_scaling:
            return

        layers_with_finite_values = self._get_layers_with_finite_values()
        for item in layers_with_finite_values:
            for param_name, param in item.layer.named_parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    logging.debug("%s scaling down %s by %s" % (item.layer_name, param_name, 1.0 / item.scaling_factor))
                    param.grad.mul_(1.0 / item.scaling_factor)

        while len(self._handles) > 0:
            elt = self._handles.pop()
            elt.remove()

    def _check_for_inf_or_nan(self) -> None:
        """
        For each layer, check if any of the parameters with a gradient attribute
        contain an inf/nan. If any of the parameters' gradient contain an inf/nan,
        then that layer's found_inf_or_nan attribute is set to True and all
        remaining parameters for that layer are skipped.
        """
        for elt in self.layer_info:
            elt.found_inf_or_nan = False
            for _, param in elt.layer.named_parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    if torch.isinf(param.grad).any().item() or torch.isnan(param.grad).any().item():  # type: ignore
                        elt.found_inf_or_nan = True
                        break  # skip all remaining named parameters

    def step(self, optimizer) -> None:  # type: ignore
        """
        If there are no inf/nan in the gradients' of all layers, then optimizer
        takes a step, otherwise not. Update the scaling factor for each layer.
        """
        # using layerwise gradient scaling
        if self._apply_layerwise_scaling:
            self._check_for_inf_or_nan()
            inf_nan_found = any(elt.found_inf_or_nan for elt in self.layer_info)

            if not inf_nan_found:
                optimizer.step()
            self._update_scale()
        # not using layerwise gradient scaling
        else:
            optimizer.step()

    def _update_scale(self) -> None:
        """
        For each layer, if an inf/nan is found, then multiply the scaling factor
        of that layer by the backoff factor and set the growth tracker of that
        layer to 0. Else, increment the growth tracker of the layer. If growth
        tracker equals the growth interval, then multiply the scaling factor of
        the layer by the growth factor and reset the layer's growth tracker to 0.
        Finally, clip the scaling factor to the range
        [self.min_scaling_factor, self.max_scaling_factor]. The min/max scaling
        factor values are user configurable.
        """
        if not self._apply_layerwise_scaling:
            return

        for layer in self.layer_info:
            if layer.found_inf_or_nan:
                if layer.scale_layer:
                    layer.scaling_factor = max(
                        self._min_scale,
                        min(self._backoff_factor * layer.scaling_factor, self._max_scale),
                    )
                    layer.growth_tracker = 0
            else:
                layer.growth_tracker += 1
                if layer.scale_layer and layer.growth_tracker == self._growth_interval:
                    layer.scaling_factor = max(
                        self._min_scale,
                        min(self._growth_factor * layer.scaling_factor, self._max_scale),
                    )
                    layer.growth_tracker = 0

    def get_layer_info(self) -> List[LayerInfo]:
        """
        Returns a list of LayerInfo instances of the model.
        """
        return self.layer_info

    def get_backward_hooks(self) -> List:
        """
        Returns a list of tuples. Each tuple contains the layer name and the
        hook attached to it.
        """
        layer_name_and_hooks = list()
        for name, layer in self._model.named_modules():
            if name != "":
                layer_name_and_hooks.append((name, layer._get_backward_hooks()))
        return layer_name_and_hooks
