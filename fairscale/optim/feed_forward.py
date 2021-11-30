import logging

import numpy as np
from sklearn.datasets import make_blobs
import torch

from fairscale.optim.grad_scaler import ShardedGradScaler


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        final_output = self.sigmoid(output)
        return final_output


def blob_label(y, label, loc):  # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


def load_data():
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


def standard_training(data, model):
    # model.reset_parameters()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
    model.train()
    num_epochs = 2
    for _ in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        logging.info(loss)

        # Backward pass
        loss.backward()
        optimizer.step()
    return model


def scale(loss, scaling_factor):
    loss *= scaling_factor
    return loss


def unscale(optimizer):
    sgs = ShardedGradScaler()

    with torch.no_grad():
        to_unscale = list()
        for group in optimizer.param_groups:
            for param in group["params"]:
                to_unscale.append(param.grad)

    found_inf = torch.Tensor([0.0])
    inv_scale = torch.Tensor([1.0 / 2 ** 13])
    sgs._foreach_non_finite_check_and_unscale_cpu_(to_unscale, found_inf, inv_scale)


def training_with_gradient_scaling(data, model2):
    # model2 = Feedforward(2, 10)
    # model.reset_parameters()
    criterion2 = torch.nn.BCELoss()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001)
    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]

    model2.train()
    num_epochs = 2
    for _ in range(num_epochs):
        optimizer2.zero_grad()
        # Forward pass
        y_pred = model2(x_train)

        # Compute Loss
        loss2 = criterion2(y_pred.squeeze(), y_train)
        logging.info("loss2 %s" % loss2)
        # logging.info("loss before scaling %s" % loss.data)

        # # Scale the loss
        scaling_factor = 2 ** 13
        # scaled_loss = scale(loss, scaling_factor)
        # logging.info("loss after scaling %s" % loss.data)

        # Backward pass
        # scaled_loss.backward()
        loss2.backward()
        # unscale weight gradient before weight update

        # unscale(optimizer)
        optimizer2.step()
        # print("fc1 weight is %s" % model.fc1.weight)
        # print("fc1 weight gradient is %s" % model.fc1.weight.grad)
        # print("fc1 bias gradient is %s" % model.fc1.bias.grad)
        # print("fc2 weight is %s" % model.fc2.weight)
        # print("fc2 weight gradient is %s" % model.fc2.weight.grad)
        # print("fc2 bias gradient is %s" % model.fc2.bias.grad)

    return model2


def reset_params(model):
    for module in model.modules():
        # print(type(module))
        if isinstance(module, Feedforward):
            for layer in module.children():
                if hasattr(layer, "reset_parameters"):
                    print("in ff %s" % layer)
                    layer.reset_parameters()
        else:
            # if isinstance(module, torch.nn.modules.linear.Linear):
            if hasattr(module, "reset_parameters"):
                print("in other %s" % module)
                module.reset_parameters()


def test_gradient_scaling():
    all_data = load_data()
    ff1 = Feedforward(2, 10)
    # reset_params(model1)
    vanilla_model = standard_training(all_data, ff1)
    logging.error(vanilla_model.fc1.bias.grad)

    ff2 = Feedforward(2, 10)
    # reset_params(model1)
    scaled_model = training_with_gradient_scaling(all_data, ff2)
    logging.error(scaled_model.fc1.bias.grad)
    # assert torch.isclose(vanilla_model.fc2.bias.grad, scaled_model.fc2.bias.grad)
