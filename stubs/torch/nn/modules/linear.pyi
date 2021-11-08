# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .module import Module
from .. import Parameter
from ... import Tensor

import torch
from typing import Union


class Identity(Module):

    def __init__(self) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Linear(Module):
    in_features: int = ...
    out_features: int = ...
    weight: Parameter = ...
    bias: Parameter = ...

    def __init__(self, in_features: int, out_features: int, bias: bool = ..., device:str = ..., dtype:Union[str, torch.dtype] = ...) -> None: ...

    def reset_parameters(self) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Bilinear(Module):
    in1_features: int = ...
    in2_features: int = ...
    out_features: int = ...
    weight: Parameter = ...
    bias: Parameter = ...

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = ...) -> None: ...

    def reset_parameters(self) -> None: ...

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor: ...  # type: ignore
