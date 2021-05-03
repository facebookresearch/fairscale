# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing scaler
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fairscale.experimental.optim.dynamic_loss_scaler import DynamicLossScaler


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


device = "cuda" if torch.cuda.is_available() else "cpu"


def _init_dataset():
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)
    # Shuffles the indices
    idx = np.arange(100)
    np.random.shuffle(idx)
    # Generates train sets
    x_train, y_train = x[idx], y[idx]
    x_train_tensor = torch.tensor([x_train]).float().to(device)
    y_train_tensor = torch.tensor([y_train]).float().to(device)
    return x_train_tensor, y_train_tensor


def _train_with_dls(x, y):
    scaler = DynamicLossScaler()
    torch.manual_seed(42)
    lr = 1e-1
    n_epochs = 1000
    loss_fn = nn.MSELoss(reduction="mean")
    model = ManualLinearRegression().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return model


def test_dls_without_overflow():
    x, y = _init_dataset()
    model = _train_with_dls(x, y)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            if name == "linear.weight":
                assert (param.data.item() - 2) <= 0.05
            if name == "linear.bias":
                assert (param.data.item() - 1) <= 0.03


# TODO(tmarkstrum): add test case covering check_overflow function
# TODO(tmarkstrum): add test case covering the state_dict, FP16
