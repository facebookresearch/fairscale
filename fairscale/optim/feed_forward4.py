import logging
from typing import List, Tuple

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
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


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

            logging.debug(
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


# assign labels
def blob_label(y: np.ndarray, label: int, loc: List) -> np.ndarray:
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


def standard_training(model: Feedforward, scaler: LayerwiseGradientScaler = None) -> Feedforward:
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x_train, y_train, x_test, y_test = load_data()
    model.train()
    num_epochs = 2
    for _ in range(num_epochs):
        optimizer.zero_grad()

        # forward pass
        y_pred = model(x_train)

        # compute loss
        loss = criterion(y_pred.squeeze(), y_train)

        # backward pass
        loss.backward()

        # unscale the weights if scaler is not None
        if scaler is not None:
            scaler.unscale()

        # update weights
        optimizer.step()
    return model

def test_feed_forward_network_parity() -> None:
    model1 = Feedforward(2, 10)
    model2 = Feedforward(2, 10)

    """
    Caveats:
    - Need to specify the linear layers and the layer after the last linear layer
    in the model since the "inputs" in the sigmoid layer need to be scaled up.
    - layers_to_scale list will be populated in the order in which the graph is
    traversed when using named_modules. This should be taken into consideration
    when initializing the scaling_factors list.
    """

    # list the scaling factors in the following order: fc1, fc2, sigmoid
    layers_to_scale = []
    scaling_factors = [32, 1024, 1]
    for name, layer in model2.named_modules():
        if isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Linear):
            layers_to_scale.append([name, layer])

    assert len(scaling_factors) == len(layers_to_scale)
    layerwise_scaler = LayerwiseGradientScaler()
    layerwise_scaler.scale(layers_to_scale, scaling_factors)

    vanilla_model = standard_training(model1)
    scaled_model = standard_training(model2, layerwise_scaler)

    # for name, layer in model2.named_modules():
    #     print(name, layer._get_backward_hooks())

    assert torch.equal(vanilla_model.fc1.weight.grad, scaled_model.fc1.weight.grad)
    assert torch.equal(vanilla_model.fc2.weight.grad, scaled_model.fc2.weight.grad)
    assert torch.equal(vanilla_model.fc1.bias.grad, scaled_model.fc1.bias.grad)
    assert torch.equal(vanilla_model.fc2.bias.grad, scaled_model.fc2.bias.grad)
