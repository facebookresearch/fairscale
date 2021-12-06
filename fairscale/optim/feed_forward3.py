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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


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


class LayerwiseGradientScaler:
    def __init__(self, scaling_factors: List):
        self.scaling_factors = scaling_factors
        # self.model = model
        self.scale_up_functions: List = []
        self.scale_down_functions: List = []

    # create a scale down and scale up function for each layer
    def make_scaling_functions(self) -> None:
        def scale_down(m: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
            if outputs[0] is not None:
                # logging.info("item value in scale down is %s", item)
                torch.mul(outputs[0], 1 / item)

        def scale_up(m: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
            if inputs[0] is not None:
                torch.mul(inputs[0], item)

        for item in self.scaling_factors:
            self.scale_up_functions.append(scale_up)
            self.scale_down_functions.append(scale_down)

    def create_scaled_functions(self) -> None:
        self.make_scaling_functions()

    def register_scaling_functions(self, layers_to_scale: List) -> None:
        for idx, layer in enumerate(layers_to_scale):
            layer[1].register_full_backward_hook(self.scale_up_functions[idx])
            layer[1].register_full_backward_hook(self.scale_down_functions[idx])


def test_parity() -> None:
    model1 = Feedforward(2, 10)
    model2 = Feedforward(2, 10)
    layers_to_scale = []

    for layer in model2.named_children():
        if isinstance(layer[1], nn.Linear):
            layers_to_scale.append(layer)

    scaling_factors = [2 ** 10, 2 ** 7]

    assert len(scaling_factors) == len(layers_to_scale)
    lgs = LayerwiseGradientScaler(scaling_factors)
    lgs.create_scaled_functions()
    lgs.register_scaling_functions(layers_to_scale)

    vanilla_model = standard_training(model1)
    scaled_model = standard_training(model2)

    assert torch.equal(vanilla_model.fc1.weight.grad, scaled_model.fc1.weight.grad)
    assert torch.equal(vanilla_model.fc2.weight.grad, scaled_model.fc2.weight.grad)
    assert torch.equal(vanilla_model.fc1.bias.grad, scaled_model.fc1.bias.grad)
    assert torch.equal(vanilla_model.fc2.bias.grad, scaled_model.fc2.bias.grad)
