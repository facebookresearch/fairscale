# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .. import Tensor
import builtins

class Parameter(Tensor):
    # These are dynamic attributes added by shard_params_data_parallel class.
    # Added here for better type checking.
    _is_sharded: bool
    _orig_size: int
    _cpu_grad: Parameter
    _full_param: Parameter
    _fp32_shard: Parameter
    _fp16_shard: Parameter

    def __init__(self, data: Tensor, requires_grad: builtins.bool = True): ...

    ...
