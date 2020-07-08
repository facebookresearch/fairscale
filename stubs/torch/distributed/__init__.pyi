# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any
from torch import Tensor

def get_rank(group: Any) -> int: ...

def get_world_size(group: Any) -> int: ...

def broadcast(tensor: Tensor, src: Any, group: Any, async_op: Any = False): ...

class group(object):
    WORLD: Any
