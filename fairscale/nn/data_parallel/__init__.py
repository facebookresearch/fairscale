# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch.distributed as dist

from .fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    TrainingState,
    auto_wrap_bn,
    get_fsdp_instances,
    no_pre_load_state_dict_hook,
)

if dist.is_available():
    # Prevent import failure if dist is not available. #1057
    from .sharded_ddp import ShardedDataParallel

__all__: List[str] = []
