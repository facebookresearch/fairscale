# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch.distributed as dist

from .checkpoint import checkpoint_wrapper
from .data_parallel import FullyShardedDataParallel

if dist.is_available():
    # Prevent import failure if dist is not available. #1057
    from .data_parallel import ShardedDataParallel
    from .moe import MOELayer, Top2Gate
    from .pipe import Pipe, PipeRPCWrapper

from .misc import FlattenParamsWrapper
from .wrap import auto_wrap, config_auto_wrap_policy, default_auto_wrap_policy, enable_wrap, wrap

__all__: List[str] = []
