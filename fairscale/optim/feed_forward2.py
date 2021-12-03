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


class Feedforward2(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(Feedforward2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

        # self.fc1.register_full_backward_hook(self.scale_up)
        self.fc1.register_full_backward_hook(self.scale_down)
        self.fc2.register_full_backward_hook(self.scale_up)
        self.fc2.register_full_backward_hook(self.scale_down)
        self.sigmoid.register_full_backward_hook(self.scale_up)
        self.sigmoid.register_full_backward_hook(self.scale_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        hidden1 = self.fc1(x)
        output1 = self.relu(hidden1)
        hidden2 = self.fc2(output1)
        final_output = self.sigmoid(hidden2)
        return final_output

    """
    multiply the output grads on layer i by S using scale_up_hook
    compute the derivative
    multiply the input grads on layer i-1 by 1/S using scale_down_hook
    """

    def scale_up(self, m, inputs, outputs):
        torch.mul(inputs[0], (2 ** 13))

    def scale_down(self, m, inputs, outputs):
        torch.mul(outputs[0], 1 / (2 ** 13))

    # def print_grads(self, m, grad_input, grad_output):
    #     print('Inside ' + self.__class__.__name__ + ' backward')
    #     print('Inside class:' + self.__class__.__name__)
    #     print('Inside module:' + self.__module__)
    #     print('')
    #     print('grad_input: ', type(grad_input), len(grad_input))
    #     print('grad_input[0]: ', type(grad_input[0]))
    #     print('grad_output: ', type(grad_output), len(grad_output))
    #     print('grad_output[0]: ', type(grad_output[0]))
    #     print('')
    #     print('grad_input size:', grad_input[0].size() if grad_input[0] is not None else None)
    #     print('grad_output size:', grad_output[0].size() if grad_input[0] is not None else None)
    #     print('grad_input norm:', grad_input[0].norm() if grad_input[0] is not None else None)


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


def standard_training(model) -> Feedforward:
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

        # compute Loss
        loss = criterion(y_pred.squeeze(), y_train)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
    return model


def test_weight() -> None:
    model1 = Feedforward(2, 10)
    model2 = Feedforward2(2, 10)
    vanilla_model = standard_training(model1)
    scaled_model = standard_training(model2)

    assert torch.equal(vanilla_model.fc1.weight.grad, scaled_model.fc1.weight.grad)
    assert torch.equal(vanilla_model.fc2.weight.grad, scaled_model.fc2.weight.grad)
    # assert torch.equal(vanilla_model.sigmoid.bias.grad, scaled_model.sigmoid.bias.grad)


def test_bias() -> None:
    model1 = Feedforward(2, 10)
    model2 = Feedforward2(2, 10)
    vanilla_model = standard_training(model1)
    scaled_model = standard_training(model2)

    assert torch.equal(vanilla_model.fc1.bias.grad, scaled_model.fc1.bias.grad)
    assert torch.equal(vanilla_model.fc2.bias.grad, scaled_model.fc2.bias.grad)
