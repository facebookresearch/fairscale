import logging
import os
from typing import Any, List, Tuple, Union

import numpy as np
from sklearn.datasets import make_blobs
import torch
from torch.cuda.amp.autocast_mode import autocast
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fairscale.optim.layerwise_gradient_scaler import LayerwiseGradientScaler


# Test 1: a simple feed forward network
class Feedforward(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = self.identity(out)
        return out


# assign labels
def blob_label(y: np.ndarray, label: int, loc: List) -> np.ndarray:
    target = np.copy(y)  # type: ignore
    for l in loc:
        target[y == l] = label
    return target


def load_data(model_type: str) -> Union[DataLoader, Tuple[Any, Any]]:
    data = None
    if model_type == "linear_model":
        torch.manual_seed(11)
        x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True, random_state=10)
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
        y_train = torch.FloatTensor(blob_label(y_train, 1, [1, 2, 3]))

        # x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=true, random_state=10)
        # x_test = torch.FloatTensor(x_test)
        # y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
        # y_test = torch.FloatTensor(blob_label(y_test, 1, [1, 2, 3]))

        data = (x_train, y_train)

    if model_type == "vision_model":
        torch.manual_seed(10)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)

        # test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        # test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

        image, _ = train_ds[0]
        assert image.shape == torch.Size([3, 32, 32])
        data = train_ds_loader  # type: ignore
    return data


def standard_training(model: Feedforward, per_layer_scaling=False) -> Feedforward:
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x_train, y_train = load_data("linear_model")

    num_epochs = 2
    model.train()

    layers_to_scale = {"fc1": 1024, "fc2": 512, "fc3": 1024} if per_layer_scaling else {}
    layer_scaler = LayerwiseGradientScaler(model, layers_to_scale)

    for _ in range(num_epochs):
        optimizer.zero_grad()

        # scale the gradients
        layer_scaler.scale()

        with autocast():
            # forward pass
            y_pred = model(x_train)
            # compute loss
            loss = criterion(y_pred.squeeze(), y_train)

        loss.backward()

        # unscale the gradients
        layer_scaler.unscale()

        # update weights and scaling factor
        layer_scaler.step(optimizer, 0)

    return model


def test_linear_model() -> None:
    model1 = Feedforward(2, 10)
    model2 = Feedforward(2, 10)

    vanilla_model = standard_training(model1, False)
    scaled_model = standard_training(model2, True)

    # to get all the backward hooks for a layer
    # for name, layer in scaled_model.named_modules():
    #     print(name, layer._get_backward_hooks())

    def get_params_with_grad(trained_model):
        result = []
        for module_name, layer in trained_model.named_modules():
            if module_name != "":
                for param_name, param in layer.named_parameters():
                    if hasattr(param, "grad"):
                        logging.debug("testing equality for %s.%s" % (module_name, param_name))
                        result.append(param.grad)
        return result

    for elt in zip(get_params_with_grad(vanilla_model), get_params_with_grad(scaled_model)):
        assert torch.allclose(elt[0], elt[1])


# Test 2: a vision model
class SimpleConvNet(nn.Module):
    def __init__(self):
        torch.manual_seed(24)
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.identity = nn.Identity()

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
        out = self.identity(out)
        return out

def vision_training(model: SimpleConvNet, per_layer_scaling=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.is_available():
        model.cuda()

    train_ds_loader = load_data("vision_model")
    model.train()

    layer_scale_dict = {"conv1": 128, "conv2": 256, "fc1": 512, "fc2": 1024, "fc3": 8192} if per_layer_scaling else {}
    layer_scaler = LayerwiseGradientScaler(model, layer_scale_dict)

    for _ in range(2):
        count = 0
        for img, lbl in train_ds_loader:
            img = img.cuda()
            lbl = lbl.cuda()

            optimizer.zero_grad()
            layer_scaler.scale()

            predict = model(img)
            loss = loss_fn(predict, lbl)

            loss.backward()

            layer_scaler.unscale()
            layer_scaler.step(optimizer, count)
            count += 1
    return model, layer_scaler.layer_info


def test_vision_model() -> None:
    """
    to remove randomness from different sources while testing set
    torch.use_deterministic_algorithms(true) and
    set the following value in the shell
    export cublas_workspace_config=:4096:8
    """
    torch.use_deterministic_algorithms(True)  # type: ignore
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    m1 = SimpleConvNet()
    m2 = SimpleConvNet()

    vision_model, _ = vision_training(m1, False)
    scaled_vision_model, layer_info = vision_training(m2, True)

    def get_params_with_grad(trained_model):
        result = []
        for module_name, layer in trained_model.named_modules():
            if module_name != "":
                for param_name, param in layer.named_parameters():
                    if hasattr(param, "grad"):
                        logging.debug("testing equality for %s.%s" % (module_name, param_name))
                        result.append(param.grad)
        return result
    
    for elt in layer_info:
        print(elt.name, elt.scale_up)
    for elt in zip(get_params_with_grad(vision_model), get_params_with_grad(scaled_vision_model)):
        assert torch.allclose(elt[0], elt[1])
    

# Test 3: Vision model with autocast
def vision_training_with_autocast(model: SimpleConvNet, per_layer_scaling=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.is_available():
        model.cuda()

    train_ds_loader = load_data("vision_model")
    model.train()

    layer_scale_dict = {"conv1": 2**10, "conv2": 2**10, "fc1": 2**10, "fc2": 2**10, "fc3": 2**10} if per_layer_scaling else {}
    layer_scaler = LayerwiseGradientScaler(model, layer_scale_dict, growth_factor = 4)

    for _ in range(2):
        count = 0
        for img, lbl in train_ds_loader:
            img = img.cuda()
            lbl = lbl.cuda()
            count += 1
            
            optimizer.zero_grad()
            layer_scaler.scale()
            
            with autocast():
                predict = model(img)
                assert predict.dtype is torch.float16
                
                loss = loss_fn(predict, lbl)
                scaled_loss = torch.mul(loss, 500) 
                assert loss.dtype is torch.float32

            scaled_loss.backward()
            layer_scaler.unscale()
            layer_scaler.step(optimizer, count)

    return model, layer_scaler.layer_info


def test_vision_model_with_autocast():
    torch.use_deterministic_algorithms(True)  # type: ignore
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    m1 = SimpleConvNet()
    m2 = SimpleConvNet()

    #vision_model, _ = vision_training_with_autocast(m1, False)
    scaled_vision_model, layer_info = vision_training_with_autocast(m2, True)

    def get_params_with_grad(trained_model):
        result = []
        for module_name, layer in trained_model.named_modules():
            if module_name != "":
                for param_name, param in layer.named_parameters():
                    if hasattr(param, "grad"):
                        logging.debug("testing equality for %s.%s" % (module_name, param_name))
                        result.append(param.grad)
        return result

    for elt in layer_info:
        print(elt.name, elt.scale_up)
    #for elt in zip(get_params_with_grad(vision_model), get_params_with_grad(scaled_vision_model)):
    #    assert torch.allclose(elt[0], elt[1])





"""
TODO:
- leave a comment that the scaling factor should be a multiple of the number of gpus.
  if there are M gpus then the scaling factor should be M \times scale.
- implement growth tracker
- write unit test for fp16, using autocast

To run the tests execute the following command from the root of the repo:
$ pytest -s tests/optim/test_layerwise_gradient_scaler.py --log-cli-level info
"""
