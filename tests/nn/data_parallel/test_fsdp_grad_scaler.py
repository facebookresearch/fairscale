# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with grad scaler. """

import os
import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn import FullyShardedDataParallel
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.utils.testing import skip_if_no_cuda

try:
    from torch.cuda.amp import autocast
except ImportError:
    # Older version doesn't support autocast. Skip this file.
    pytestmark = pytest.mark.skip


# Mixed precision needs cuda.
@skip_if_no_cuda
def test_scaler_cpu_offload_breaks():
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    # Random port in case the next test run quickly, same port would cause conflict.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(2000, 3000))
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    try:
        scaler = ShardedGradScaler()
        model = FullyShardedDataParallel(nn.Linear(5, 5), cpu_offload=True, mixed_precision=True)
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)

        input = torch.rand((1, 5), dtype=torch.float).to(device)
        optim.zero_grad()
        with autocast():
            output = model(input)
            loss = F.mse_loss(input, output)

        scaler.scale(loss).backward()
        # TODO (Min): Need to fix. Details in issue #421.
        with pytest.raises(RuntimeError):
            scaler.step(optim)
            scaler.update()

    finally:
        # Clean-up is important or the next test in this file may fail to init the PG.
        torch.distributed.destroy_process_group()
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]
