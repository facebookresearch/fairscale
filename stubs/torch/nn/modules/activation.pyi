# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from ... import Tensor
from .. import Parameter
from .module import Module
from typing import Any, Optional


class Threshold(Module):
    threshold: float = ...
    value: float = ...
    inplace: bool = ...

    def __init__(self, threshold: float, value: float, inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class ReLU(Threshold):
    def __init__(self, inplace: bool = ...) -> None: ...


class RReLU(Module):
    lower: float = ...
    upper: float = ...
    inplace: bool = ...

    def __init__(self, lower: float = ..., upper: float = ..., inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Hardtanh(Module):
    min_val: float = ...
    max_val: float = ...
    inplace: bool = ...

    def __init__(self, min_val: float = ..., max_val: float = ..., inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class ReLU6(Hardtanh):
    def __init__(self, inplace: bool = ...) -> None: ...


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class ELU(Module):
    alpha: float = ...
    inplace: bool = ...

    def __init__(self, alpha: float = ..., inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class CELU(Module):
    alpha: float = ...
    inplace: bool = ...

    def __init__(self, alpha: float = ..., inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class SELU(Module):
    inplace: bool = ...

    def __init__(self, inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class GLU(Module):
    dim: int = ...

    def __init__(self, dim: int = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class GELU(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Hardshrink(Module):
    lambd: float = ...

    def __init__(self, lambd: float = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class LeakyReLU(Module):
    negative_slope: float = ...
    inplace: bool = ...

    def __init__(self, negative_slope: float = ..., inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class LogSigmoid(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softplus(Module):
    beta: float = ...
    threshold: float = ...

    def __init__(self, beta: float = ..., threshold: float = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softshrink(Module):
    lambd: float = ...

    def __init__(self, lambd: float = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class PReLU(Module):
    num_parameters: int = ...
    weight: Parameter = ...

    def __init__(self, num_parameters: int = ..., init: float = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softsign(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Tanhshrink(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softmin(Module):
    dim: int = ...

    def __init__(self, dim: Optional[int] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softmax(Module):
    dim: int = ...

    def __init__(self, dim: Optional[int] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softmax2d(Module):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class LogSoftmax(Module):
    dim: int = ...

    def __init__(self, dim: Optional[int] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, bias: bool, add_bias_kv: bool, add_zero_attn: bool, kdim: Optional[int], vdim: Optional[int]) -> None: ...
