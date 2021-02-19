# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional
from .. import Size, Tensor
from ..cuda import Stream
import builtins

class Parameter(Tensor):
    # These are dynamic attributes added by shard_params_data_parallel class.
    # Added here for better type checking.
    _is_sharded: bool
    _orig_size: Size
    _cpu_grad: Tensor
    _full_param_padded: Tensor
    _fp32_shard: Tensor
    _fp16_shard: Optional[Tensor]

    def __init__(self, data: Tensor, requires_grad: builtins.bool = True): ...

    ...
