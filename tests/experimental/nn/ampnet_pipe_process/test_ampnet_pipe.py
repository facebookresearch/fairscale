# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from fairscale.experimental.nn.ampnet_pipe.pipe import AMPnetPipe
from fairscale.fair_dev.testing.testing import get_worker_map, torch_spawn


class MySGD(Optimizer):
    r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
    """

    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MySGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-group["lr"])
        return loss


class AMPnetDelegate(object):
    def __init__(self, vocab_size=100, iteration_per_batch=1000):
        self.iteration_per_batch = iteration_per_batch
        self.vocab_size = vocab_size

    def transform_input(self, cur_batch):
        return cur_batch["input"]

    def transform_target(self, cur_batch):
        return cur_batch["target"]

    def log_loss(self, cur_batch, loss, count):
        pass

    def transform_output_before_loss(self, output_tensor):
        return output_tensor

    def check_and_save_weights(self, num_gradients):
        pass


class FakeDataset(Dataset):
    def __init__(
        self,
        input_dim=10,
        output_dim=10,
        total_samples=100,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_samples = total_samples
        self.input_samples = torch.rand(self.total_samples, self.input_dim, self.output_dim)
        self.target_samples = torch.rand(self.total_samples, self.input_dim, self.output_dim)

    def __getitem__(self, index):
        return {
            "input": self.input_samples[index, :, :],
            "target": self.target_samples[index, :, :],
        }

    def __len__(self):
        return self.total_samples


@torch_spawn([2])
def async_event_loop_interleave_simple():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(inplace=False), nn.Linear(10, 10), nn.ReLU(inplace=False))
    pipe = AMPnetPipe(
        module=model,
        balance=[2, 2],
        worker_map=get_worker_map(),
        chunks=10,
        checkpoint="never",
    )
    fake_dataset = FakeDataset()
    fake_dataloader = DataLoader(fake_dataset, batch_size=4, shuffle=True, num_workers=0)
    loss = nn.MSELoss()
    opt = MySGD(model.parameters(), lr=0.01)
    transform_and_log = AMPnetDelegate()
    pipe.interleave(fake_dataloader, loss, opt, transform_and_log)


@torch_spawn([4])
def async_event_loop_interleave_hard():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10))
    pipe = AMPnetPipe(
        module=model,
        balance=[1, 1, 1, 1],
        worker_map=get_worker_map(),
        chunks=10,
        checkpoint="never",
    )
    fake_dataset = FakeDataset()
    fake_dataloader = DataLoader(fake_dataset, batch_size=4, shuffle=True, num_workers=0)
    loss = nn.MSELoss()
    opt = MySGD(model.parameters(), lr=0.01)
    transform_and_log = AMPnetDelegate()
    pipe.interleave(fake_dataloader, loss, opt, transform_and_log)
