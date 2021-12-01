import logging
from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs
import torch

from fairscale.optim.grad_scaler import ShardedGradScaler


class Feedforward(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        hidden1 = self.fc1(x)
        output1 = self.relu(hidden1)
        hidden2 = self.fc2(output1)
        final_output = self.sigmoid(hidden2)
        return final_output


# assign labels
def blob_label(y: np.ndarray, label: int, loc: list) -> np.ndarray:
    target = np.copy(y)
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


def standard_training() -> Feedforward:
    model = Feedforward(2, 10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x_train, y_train, x_test, y_test = load_data()
    model.train()
    num_epochs = 2
    for _ in range(num_epochs):
        optimizer.zero_grad()

        # forward pass
        y_pred = model(x_train)

        # compute Loss
        loss = criterion(y_pred.squeeze(), y_train)

        # backward pass
        loss.backward()
        optimizer.step()
    return model


def scale(loss: torch.Tensor, scaling_factor: int) -> torch.Tensor:
    loss *= scaling_factor
    return loss


def unscale(optimizer: torch.optim.SGD, model: Feedforward) -> None:
    sgs = ShardedGradScaler()

    # unscaling using the optimizer
    # with torch.no_grad():
    #     to_unscale = list()
    #     for group in optimizer.param_groups:
    #         for param in group["params"]:
    #             to_unscale.append(param.grad)

    #     print(to_unscale)

    # unscaling using the model params
    with torch.no_grad():
        to_unscale_fc1 = list()
        to_unscale_fc2 = list()
        for name, value in model.named_parameters():
            if "fc1" in name:
                to_unscale_fc1.append(value.grad)
            else:
                to_unscale_fc2.append(value.grad)
        logging.info(to_unscale_fc1)
        logging.info(to_unscale_fc2)

    found_inf = torch.Tensor([0.0])
    inv_scale1 = torch.Tensor([1.0 / 2 ** 13])
    inv_scale2 = torch.Tensor([1.0 / 2 ** 13])
    sgs._foreach_non_finite_check_and_unscale_cpu_(to_unscale_fc1, found_inf, inv_scale1)
    sgs._foreach_non_finite_check_and_unscale_cpu_(to_unscale_fc2, found_inf, inv_scale2)


def training_with_gradient_scaling() -> Feedforward:
    model = Feedforward(2, 10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    x_train, y_train, x_test, y_test = load_data()

    model.train()
    num_epochs = 2
    for _ in range(num_epochs):
        optimizer.zero_grad()

        # forward pass
        y_pred = model(x_train)

        # compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        # scale the loss
        scaling_factor = 2 ** 13
        scaled_loss = scale(loss, scaling_factor)

        # backward pass
        scaled_loss.backward()

        # unscale weight gradient before weight update
        unscale(optimizer, model)
        optimizer.step()
    return model


def test_gradient_scaling() -> None:
    vanilla_model = standard_training()
    scaled_model = training_with_gradient_scaling()

    assert torch.equal(vanilla_model.fc1.bias, scaled_model.fc1.bias)
    assert torch.equal(vanilla_model.fc2.bias, scaled_model.fc2.bias)
    assert torch.equal(vanilla_model.fc1.bias.grad, scaled_model.fc1.bias.grad)
    assert torch.equal(vanilla_model.fc2.bias.grad, scaled_model.fc2.bias.grad)

    assert torch.equal(vanilla_model.fc1.weight, scaled_model.fc1.weight)
    assert torch.equal(vanilla_model.fc2.weight, scaled_model.fc2.weight)
    assert torch.equal(vanilla_model.fc1.weight.grad, scaled_model.fc1.weight.grad)
    assert torch.equal(vanilla_model.fc2.weight.grad, scaled_model.fc2.weight.grad)
