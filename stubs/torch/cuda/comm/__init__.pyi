# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#MODIFIED BY TORCHGPIPE
from typing import Iterable, Optional, Tuple

from torch import Tensor


def scatter(tensor: Tensor,
            devices: Iterable[int],
            chunk_sizes: Optional[Iterable[int]] = None,
            dim: int = 0,
            ) -> Tuple[Tensor, ...]: ...


def gather(tensors: Iterable[Tensor],
           dim: int = 0,
           destination: Optional[int] = None,
           ) -> Tensor: ...


def broadcast_coalesced(tensors: Iterable[Tensor],
           devices: Iterable[int],
           buffer_size: int = 10485760,
           ) -> Tuple[Tensor, ...]: ...


def reduce_add_coalesced(inputs: Iterable[Iterable[Tensor]],
           destination: Optional[int] = None,
           buffer_size: int = 10485760,
           ) -> Tuple[Tensor, ...]: ...

#END
