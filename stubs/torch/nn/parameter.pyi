# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional, Tuple, Any
from .. import Size, Tensor
from ..cuda import Stream
import builtins

class Parameter(Tensor):
    # These are dynamic attributes added by shard_params_data_parallel class.
    # Added here for better type checking.
    _is_sharded: bool
    _is_shared: bool
    _orig_size: Size
    _cpu_grad: Tensor
    _full_param_padded: Tensor
    _fp32_shard: Tensor
    _fp16_shard: Optional[Tensor]
    _shard_bwd_hook: Tuple[Any, Any]
    _saved_grad_shard: Tensor
    _linked_param: Parameter

    def __new__(cls, data: Tensor, requires_grad: builtins.bool = True): ...

    def __init__(self, data: Tensor, requires_grad: builtins.bool = True): ...

    ...
