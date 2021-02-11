# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from . import Tensor
from typing import Tuple, List, Union, Optional, Any


def split(tensor: Tensor, split_size_or_sections: Union[int, List[int]], dim: int=0) -> Tuple[Tensor,...]: ...

def einsum(equation: str, *operands: Tensor): ...

def norm(input: Tensor, p: Union[int, float, Any], dim: Optional[List[int]]=None, keep_dim: Optional[bool]=False, out: Optional[Tensor]=None, dtype:Optional[int]=None) -> Tensor : ...
