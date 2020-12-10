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

"""A helper to roughly balance a sequential module.

Usage::

    import torch
    from fairscale.nn import Pipe
    from fairscale.nn.pipe.balance import balance_by_time

    sample = torch.empty(128, 3, 224, 224)
    balance = balance_by_time(torch.cuda.device_count(), model, sample)

    pipe = Pipe(model, balance, chunks=8)

.. note::
    balance_by_time does not work with inplace ReLU because we exhausetively search
    every partition boundary, which could hit an inplace ReLU.

.. note::
    If the model is larger than a single CUDA device memory, use "cpu"
    in the balance_by_time function.
"""
from typing import List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from . import blockpartition
from .profile import profile_sizes, profile_times

__all__ = ["balance_by_time", "balance_by_size"]


Device = Union[torch.device, int, str]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def _balance_cost(cost: List[int], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(cost, partitions)
    return [len(p) for p in partitioned]


def balance_by_time(
    partitions: int,
    module: nn.Sequential,
    sample: TensorOrTensors,
    *,
    timeout: float = 1.0,
    device: Device = torch.device("cuda"),
) -> List[int]:
    """Naive automatic balancing by elapsed time per layer.
    ::

        sample = torch.empty(128, 3, 224, 224)
        balance = balance_by_time(torch.cuda.device_count(), model, sample)
        pipe = Pipe(model, balance, chunks=8)

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        sample (torch.Tensor):
            example input with arbitrary batch size

    Keyword Args:
        timeout (float):
            profiling iterates again if the timeout (in second) is not exceeded
            (default: ``1.0``)
        device ('cpu' or 'cuda' device):
            CPU or CUDA device where each layer is profiled (default: the
            current CUDA device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~fairscale.nn.Pipe`.

    .. note::
        `module` and `sample` must be placed on the same device.

    """
    times = profile_times(module, sample, timeout, torch.device(device))
    return _balance_cost(times, partitions)


def balance_by_size(
    partitions: int,
    module: nn.Sequential,
    input: TensorOrTensors,
    *,
    chunks: int = 1,
    param_scale: float = 2.0,
    device: Device = torch.device("cuda"),
) -> List[int]:
    """Naive automatic balancing by CUDA memory usage per layer.

    During training, required memory for parameters depends on which optimizer
    is used. Optimizers may use buffers for each parameter to track
    optimization statistics internally, such as momentum buffer in SGD.

    To get more reliable size based balance, you should specify `param_scale`
    with regard to your optimizer. The default `param_scale` is 2 instead of 1
    due to gradient accumulation which is necessary for every optimizer.

    Follow this guide to choose correct `param_scale` for typical optimizers:

    =========  =============  =========================================
    Optimizer  `param_scale`  Internal State
    =========  =============  =========================================
    SGD        2--3           (momentum_buffer)
    Adam       4--5           exp_avg, exp_avg_sq, (max_exp_avg_sq)
    Adadelta   4              square_avg, acc_delta
    Adagrad    3              sum
    RMSprop    3--5           square_avg, (momentum_buffer), (grad_avg)
    =========  =============  =========================================

    Here's a simple example with the Adam optimizer::

        balance = balance_by_size(
            torch.cuda.device_count(),
            model,

            # Same size with mini-batch to train
            torch.empty(1024, 3, 224, 224),

            # Number of micro-batches to train with Pipe
            chunks=8,

            # 4 for Adam
            param_scale=4.0,
        )

        pipe = Pipe(model, balance, chunks=8)
        adam = Adam(pipe.parameters())

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        input (torch.Tensor):
            example mini-batch with the same size to train

    Keyword Args:
        chunks (int):
            number of micro-batches will be used to train (default: ``1``)
        param_scale (float):
            how many copies of parameters would be allocated for training. It
            depends on optimizer. See the above guide. (default: ``2.0``)
        device ('cuda' device):
            CUDA device where each layer is profiled (default: the current CUDA
            device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~fairscale.nn.Pipe`.

    .. note::
        `module` and `input` must be placed on the same CUDA device.

    """
    sizes = profile_sizes(module, input, chunks, param_scale, torch.device(device))
    return _balance_cost(sizes, partitions)
