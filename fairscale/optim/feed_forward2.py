import logging
from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn


class Feedforward(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class LayerwiseGradientScaler:
    def __init__(self, modules: nn.Module):
        self.per_layer_scale = [2 ** (1 / 1000) for _ in modules]
        # print('length of modules %s' % len(modules))

    def scale_up(self, m: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        if inputs[0] is not None:
            torch.mul(inputs[0], (2 ** 13))

    def scale_down(self, m: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        if outputs[0] is not None:
            torch.mul(outputs[0], 1 / (2 ** 13))


# assign labels
def blob_label(y: np.ndarray, label: int, loc: list) -> np.ndarray:
    target = np.copy(y)  # type: ignore
    for l in loc:
        target[y == l] = label
    return target


def load_data() -> Tuple:
    torch.manual_seed(11)
    x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True, random_state=10)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
    y_train = torch.FloatTensor(blob_label(y_train, 1, [1, 2, 3]))

    x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True, random_state=10)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
    y_test = torch.FloatTensor(blob_label(y_test, 1, [1, 2, 3]))

    all_data = (x_train, y_train, x_test, y_test)
    return all_data


def standard_training(model: Feedforward) -> Feedforward:
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x_train, y_train, x_test, y_test = load_data()
    model.train()
    num_epochs = 2
    for _ in range(num_epochs):
        optimizer.zero_grad()

        # forward pass
        y_pred = model(x_train)
        # print(y_pred)

        # compute loss
        loss = criterion(y_pred.squeeze(), y_train)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
    return model


def test_weights() -> None:
    model1 = Feedforward(2, 10)
    model2 = Feedforward(2, 10)

    layers_to_scale = list()
    for layer in model2.named_children():
        print(layer)
        if isinstance(layer[1], nn.Linear):
            layers_to_scale.append(layer)

    layerwise_grad_scaler = LayerwiseGradientScaler(layers_to_scale)
    for _, layer in enumerate(layers_to_scale):
        layer[1].register_full_backward_hook(layerwise_grad_scaler.scale_up)
        layer[1].register_full_backward_hook(layerwise_grad_scaler.scale_down)

    vanilla_model = standard_training(model1)
    scaled_model = standard_training(model2)

    logging.info([item for item in vanilla_model.sigmoid.parameters()])
    assert torch.equal(vanilla_model.fc1.weight.grad, scaled_model.fc1.weight.grad)
    assert torch.equal(vanilla_model.fc2.weight.grad, scaled_model.fc2.weight.grad)
    assert torch.equal(vanilla_model.fc1.bias.grad, scaled_model.fc1.bias.grad)
    assert torch.equal(vanilla_model.fc2.bias.grad, scaled_model.fc2.bias.grad)


# class LayerwiseGradScaler(AbstractLayerwiseGradientScaler):
#     def __init__(self, model):
#         self.layers_to_scale = list()
#         self.scale_up_functions = list()
#         self.scale_down_functions = list()
#         self.scaling_factors = [ 2**7, 2**15 ]
#         self.make_scaling_functions()
#         self.register_scaling_functions()

#         for layer in model.named_children():
#             if isinstance(layer[1], nn.Linear):
#                 self.layers_to_scale.append(layer)

#         assert len(self.scaling_factors) == len(self.layers_to_scale)

# def get_scaled_model(model):
#     layers_to_scale = list()
#     scale_up_functions = list()
#     scale_down_functions = list()
#     scaling_factors = [ 2**7, 2**15 ]

#     # create a scale down and scale up function for each layer
#     def make_scaling_functions():
#         def scale_down(self, m, inputs, outputs):
#             if outputs[0] is not None:
#                 return torch.mul(outputs[0], item)

#         def scale_up(self, m, inputs, outputs):
#             if inputs[0] is not None:
#                 return torch.mul(inputs[0], item)

#         for item in scaling_factors:
#             print(item)
#             scale_up_functions.append(scale_up)
#             scale_down_functions.append(scale_down)
#         print(scale_up_functions)
#         print(scale_down_functions)

#     # register the scale up and scale down functions for each layer
#     def register_scaling_functions():
#         for idx, layer in enumerate(layers_to_scale):
#             print(idx, layer[0], layer[1])
#             layer[1].register_full_backward_hook(scale_up_functions[idx])
#             layer[1].register_full_backward_hook(scale_down_functions[idx])

#     for layer in model.named_children():
#         if isinstance(layer[1], nn.Linear):
#             layers_to_scale.append(layer)
#     make_scaling_functions()
#     print(layers_to_scale)
#     register_scaling_functions()
#     # print([item for item in model._get_backward_hooks()])
#     return model

"""
Idea:
Create scale_up[i] and scale_down[i] functions for each layer of the model.
Bind a unique constant to each of these functions and then pass them to the
register_full_backward_hook function.

The class should take a list of scaling factors and return a list of functions
that can be registered with the register_full_backward_hook function.

Design:

FeedForward class
    - implements the network and forward function

LayerwiseGradientScaler
    - input:
        - instance of a model
        - list of scaling factors
    - implements a function which returns the list of scaling functions on which we will
    be passed to the backward hook
    - output:
        - list of scaling functions

Main
    - create an instance of the model
    - generate a list of layers on which scaling needs to be applied
    - specify a list of scaling factors
    - pass the list of scaling factors to LayerwiseGradientScaler
    - loop through the layers and register the backward hook on the model

Test
    - compare the gradients of fc1 and fc2 of the vanilla model with the
      model returned from main
"""
