# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from . import Tensor
from typing import Tuple, List, Union

def split(tensor: Tensor, split_size_or_sections: Union[int, List[int]], dim: int=0) -> Tuple[Tensor,...]: ...

def einsum(equation: str, *operands: Tensor): ...

