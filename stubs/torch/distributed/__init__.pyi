# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, List, Union, Optional
from torch import Tensor
import datetime

class Backend: ...
class ProcessGroup: ...

class ReduceOp:
    SUM: ReduceOp
    PRODUCT: ReduceOp
    MIN: ReduceOp
    MAX: ReduceOp
    BAND: ReduceOp
    BOR: ReduceOp
    BXOR: ReduceOp

def get_rank(group: Any = None) -> int: ...

def get_world_size(group: Any = None) -> int: ...

def broadcast(tensor: Tensor, src: Any, group: Any, async_op: Any = False): ...

def is_initialized() -> bool: ...

def new_group(ranks: List[int], timeout: datetime.timedelta = datetime.timedelta(0, 1800), backend: Union[None, str, Backend] = None): ...

def all_reduce(tensor: Tensor, op: ReduceOp = ReduceOp.SUM, group:Optional[ProcessGroup] = None, async_op: bool = False): ...
def all_gather(tensor_list: List[Tensor], tensor: Tensor, group:Optional[ProcessGroup] = None, async_op: bool = False): ...

class group(object):
    WORLD: Any
