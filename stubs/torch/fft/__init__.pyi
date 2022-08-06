# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional
from torch import Tensor

# See https://github.com/python/mypy/issues/4146 for why these workarounds
# is necessary
#_int = builtins.int
#_float = builtins.float
#_bool = builtins.bool
#_size = Union[Size, List[int], Tuple[int, ...]]


def fft(input: Tensor, n: Optional[int] = None, dim: Optional[int]=-1, norm: Optional[str]=None) -> Tensor: ...
def ifft(input: Tensor, n: Optional[int] = None, dim: Optional[int]=-1, norm: Optional[str]=None) -> Tensor: ...
def rfft(input: Tensor, n: Optional[int] = None, dim: Optional[int]=-1, norm: Optional[str]=None) -> Tensor: ...
def irfft(input: Tensor, n: Optional[int] = None, dim: Optional[int]=-1, norm: Optional[str]=None) -> Tensor: ...
