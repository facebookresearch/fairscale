# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp

def all_reduce(tensor: Tensor, op: ReduceOp = ReduceOp.SUM, group:Optional[ProcessGroup] = None): ...
