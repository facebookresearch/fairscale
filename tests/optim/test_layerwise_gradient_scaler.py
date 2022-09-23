# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, List, Tuple, Union

import numpy as np
import pytest
from sklearn.datasets import make_blobs
import torch
from torch.cuda.amp.autocast_mode import autocast
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fairscale.fair_dev.common_paths import DATASET_CACHE_ROOT
from fairscale.fair_dev.testing.testing import skip_a_test_if_in_CI
from fairscale.optim.layerwise_gradient_scaler import LayerwiseGradientScaler


# Test: feed forward network
class FeedForward(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(FeedForward, self).__init__()
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

        data = (x_train, y_train)

    if model_type == "vision_model":
        torch.manual_seed(10)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # TODO: we should NOT do this download over and over again during test.
        train_ds = torchvision.datasets.CIFAR10(
            root=DATASET_CACHE_ROOT,
            train=True,
            download=True,
            transform=transform,
        )
        train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)

        image, _ = train_ds[0]
        assert image.shape == torch.Size([3, 32, 32])
        data = train_ds_loader  # type: ignore
    return data


def get_params_with_grad(trained_model):
    result = []
    for module_name, layer in trained_model.named_modules():
        if module_name != "":
            for param_name, param in layer.named_parameters():
                if hasattr(param, "grad"):
                    logging.debug("testing equality for %s.%s" % (module_name, param_name))
                    result.append(param.grad)
    return result


def train_linear_model(model: FeedForward, per_layer_scaling=False) -> FeedForward:
    criterion = torch.nn.BCEWithLogitsLoss()
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
        layer_scaler.step(optimizer)

    return model


def test_linear_model() -> None:
    model1 = FeedForward(2, 10)
    model2 = FeedForward(2, 10)

    vanilla_model = train_linear_model(model1, False)
    scaled_model = train_linear_model(model2, True)

    for elt in zip(get_params_with_grad(vanilla_model), get_params_with_grad(scaled_model)):
        assert torch.allclose(elt[0], elt[1])


# Test: convolutional network
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


def train_vision_model(model: SimpleConvNet, per_layer_scaling=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.is_available():
        model.cuda()

    train_ds_loader = load_data("vision_model")
    model.train()

    layer_scale_dict = {"conv1": 128, "conv2": 256, "fc1": 512, "fc2": 1024, "fc3": 8192} if per_layer_scaling else {}
    layer_scaler = LayerwiseGradientScaler(model, layer_scale_dict)

    for _ in range(2):
        for img, lbl in train_ds_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                lbl = lbl.cuda()

            optimizer.zero_grad()
            layer_scaler.scale()

            predict = model(img)
            loss = loss_fn(predict, lbl)

            loss.backward()

            layer_scaler.unscale()
            layer_scaler.step(optimizer)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_vision_model() -> None:
    # The os.environ below doesn't seem to be enough if the test is run on CI with many other tests
    # together.
    # see: https://app.circleci.com/pipelines/github/facebookresearch/fairscale/4086/workflows/72b1470a-55f8-4a45-afe5-04641b093bef/jobs/45179/tests#failed-test-0
    # Skipping for now.
    # Also, TODO (Min): improving downloading code above before re-enable this.
    skip_a_test_if_in_CI()
    # Remove randomness from various sources while testing.
    torch.use_deterministic_algorithms(True)  # type: ignore
    # set environment variable in CircleCI for test to pass: CUBLAS_WORKSPACE_CONFIG = :4096:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    m1 = SimpleConvNet()
    m2 = SimpleConvNet()

    vision_model = train_vision_model(m1, False)
    scaled_vision_model = train_vision_model(m2, True)

    for elt in zip(get_params_with_grad(vision_model), get_params_with_grad(scaled_vision_model)):
        assert torch.allclose(elt[0], elt[1])
