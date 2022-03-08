# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import fairscale.experimental.nn


def benchmark_bn(rank, world_size, init_file, bn_cls):
    dist.init_process_group(dist.Backend.NCCL, init_method="file://" + init_file, rank=rank, world_size=world_size)
    x = torch.randn(50, 2048, 7, 7).to(rank)
    bn = bn_cls(2048).to(rank)
    bn = DDP(bn, device_ids=[rank])
    # Warmup
    for i in range(50):
        with torch.no_grad():
            x = bn(x)
    torch.cuda.synchronize(rank)
    t0 = time.time()
    for i in range(100):
        with torch.no_grad():
            x = bn(x)
    torch.cuda.synchronize(rank)
    t1 = time.time()
    print("Elapsed time is ", t1 - t0)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    for cls in [torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, fairscale.experimental.nn.SyncBatchNorm]:
        print(cls)
        mp.spawn(benchmark_bn, args=(world_size, tempfile.mkstemp()[1], cls), nprocs=world_size)
