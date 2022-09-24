# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from fairscale.fair_dev.testing.testing import skip_if_single_gpu, temp_files_ctx
from fairscale.nn import enable_wrap, wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def main(rank, sync_file):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{sync_file}",
        world_size=2,
        rank=rank,
    )
    ffn = FFN().cuda().half()

    with enable_wrap(wrapper_cls=FullyShardedDataParallel):
        model = wrap(
            ffn,
            process_group=torch.distributed.new_group(),
            flatten_parameters=True,
            compute_dtype=torch.float16,
        )

    model = model.train()

    # We test this behavior because it might be used by pipelining.
    # However, we don't check if the speed (compute/comm overlapping)
    # and memory (necessary all-gather & free) are optimal.
    losses = []
    for _ in range(3):
        x = torch.rand((10, 10)).cuda().half()
        out = model(x)
        loss = out.sum()
        losses.append(loss)

    # Only the last bwd can be outside of no_sync context.
    with model.no_sync():
        losses[0].backward()
        losses[1].backward()
    losses[2].backward()


@skip_if_single_gpu
def test_fwd_fwd_bwd_bwd():
    with temp_files_ctx(num=1) as temp_files:
        torch.multiprocessing.spawn(
            fn=main,
            nprocs=2,
            args=(temp_files[0],),
            join=True,
        )
