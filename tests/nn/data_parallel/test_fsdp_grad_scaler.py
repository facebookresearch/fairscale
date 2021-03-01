import os
from unittest import mock

import pytest
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn import FullyShardedDataParallel
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.utils.testing import skip_if_no_cuda


@mock.patch.dict(os.environ, {"MASTER_ADDR": "localhost", "MASTER_PORT": "1337"}, clear=True)
@skip_if_no_cuda
def test_scaler_cpu_offload():
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

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
    torch.distributed.destroy_process_group()
