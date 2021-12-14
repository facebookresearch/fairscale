import logging
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from fairscale.optim.layerwise_gradient_scaler import LayerwiseGradientScaler

# torch.backends.cudnn.benchmark = False


# Test 1: a simple feed forward network
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

    num_epochs = 2
    model.train()

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


def test_linear_model() -> None:
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

    def get_params_with_grad(trained_model):
        result = []
        for name, layer in trained_model.named_modules():
            if isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    if hasattr(param, "grad"):
                        result.append(param.grad)
        return result

    for elt in zip(get_params_with_grad(vanilla_model), get_params_with_grad(scaled_model)):
        assert torch.equal(elt[0], elt[1])


###############################################################################

# Test 2: a vision model


class SimpleConvNet(nn.Module):
    def __init__(self):
        torch.manual_seed(24)
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)  # flatten all dimensions except batch
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.relu4(out)
        out = self.fc3(out)
        return out


def load_data_for_vision():
    torch.manual_seed(10)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)

    # test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    # test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    image, label = train_ds[0]
    assert image.shape == torch.Size([3, 32, 32])
    return train_ds_loader


def vision_training(model: SimpleConvNet, scaler: LayerwiseGradientScaler = None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_ds_loader = load_data_for_vision()
    num_epochs = 1

    # if torch.cuda.is_available():
    #     model.cuda()

    model.train()
    for _ in range(num_epochs):
        for img, lbl in train_ds_loader:
            # img = img.cuda()
            # lbl = lbl.cuda()

            optimizer.zero_grad()
            predict = model(img)

            loss = loss_fn(predict, lbl)
            loss.backward()

            if scaler is not None:
                scaler.unscale()

            optimizer.step()

    return model


def test_vision_model() -> None:
    """
    requires the following value in the shell
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    """
    # required to get rid of randomness from different sources
    torch.use_deterministic_algorithms(True)

    m1 = SimpleConvNet()
    m2 = SimpleConvNet()

    layers_to_scale = []
    scaling_factors = [32, 1]

    for name, layer in m2.named_modules():
        # if isinstance(layer, nn.Conv2d):
        # print(name, layer)
        if name in ["conv2", "relu2"]:
            layers_to_scale.append([name, layer])

    print(layers_to_scale)
    assert len(scaling_factors) == len(layers_to_scale)
    layerwise_scaler = LayerwiseGradientScaler()
    layerwise_scaler.scale(layers_to_scale, scaling_factors)

    vision_model = vision_training(m1)
    scaled_vision_model = vision_training(m2, layerwise_scaler)

    # assert torch.equal(vision_model.conv1.weight.grad, scaled_vision_model.conv1.weight.grad)
    # assert torch.equal(vision_model.conv1.bias.grad, scaled_vision_model.conv1.bias.grad)

    # assert torch.equal(vision_model.conv2.weight.grad, scaled_vision_model.conv2.weight.grad)
    # assert torch.equal(vision_model.conv2.bias.grad, scaled_vision_model.conv2.bias.grad)

    # assert torch.equal(vision_model.fc1.weight.grad, scaled_vision_model.fc1.weight.grad)
    # assert torch.equal(vision_model.fc1.bias.grad, scaled_vision_model.fc1.bias.grad)

    # assert torch.equal(vision_model.fc2.weight.grad, scaled_vision_model.fc2.weight.grad)
    # assert torch.equal(vision_model.fc2.bias.grad, scaled_vision_model.fc2.bias.grad)

    # assert torch.equal(vision_model.fc3.weight.grad, scaled_vision_model.fc3.weight.grad)
    # assert torch.equal(vision_model.fc3.bias.grad, scaled_vision_model.fc3.bias.grad)

    def get_params_with_grad(trained_model):
        result = []
        for name, layer in trained_model.named_modules():
            if name != "":
                for param_name, param in layer.named_parameters():
                    if hasattr(param, "grad"):
                        logging.info("testing equality for %s.%s" % (name, param_name))
                        result.append(param.grad)
        return result

    for elt in zip(get_params_with_grad(vision_model), get_params_with_grad(scaled_vision_model)):
        assert torch.equal(elt[0], elt[1])

    """
    IMPORTANT: in the shell, export CUBLAS_WORKSPACE_CONFIG=:4096:8 to get rid of randomness from different sources.

    scaling and unscaling should be done taking the device into account. at present the scaling
    and unscaling functions do not take the device into account.
    """
