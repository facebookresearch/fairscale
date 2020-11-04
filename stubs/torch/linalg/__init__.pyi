import torch
from math import inf
from typing import Optional, Union


def norm(
    input: torch.Tensor,
    ord=Union[int, float, str, inf],
    dim: Union[int, tuple, Optional[int]] = None,
    keepdim: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    ...
