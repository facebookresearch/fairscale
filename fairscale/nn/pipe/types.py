# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

ACTIVATIONS_GRADS_QUEUE = 0
SKIP_TENSOR_QUEUE = 1
PORTAL_QUEUE = 2
EVENT_LOOP_QUEUE = 3
EVENT_LOOP_ACTIVATIONS_QUEUE = 4
EVENT_LOOP_GRADIENTS_QUEUE = 5
MESSAGE_GENERATION_START = 6

MessageGeneration = MESSAGE_GENERATION_START

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

InputDevice = Union[None, int, str, torch.device]


class LazyModule:
    def __init__(self, function: Callable[[], nn.Module]):
        self.function = function

    def __call__(self) -> nn.Module:
        return self.function()


@dataclass(init=False)
class PipeMessage:
    src: int
    dest: int
    queue_name: int
    args: Any
    tensors: Tensors
    tensor_shapes: List[torch.Size]
    tensor_dtypes: List[torch.dtype]
    tag: int = 0

    def __init__(
        self,
        src: int,
        dest: int,
        queue_name: int,
        args: Any = None,
        tensors: Optional[Tensors] = None,
        tensor_count: int = 0,
    ):
        self.src = src
        self.dest = dest
        self.queue_name = queue_name
        self.args = args
        self.tensors = tensors or tuple()
        self.tensor_shapes = []
        self.tensor_dtypes = []

        global MessageGeneration
        self.tag = MessageGeneration
        if tensors is None:
            MessageGeneration += tensor_count
        else:
            MessageGeneration += len(self.tensors)
