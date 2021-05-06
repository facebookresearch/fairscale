# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.experimental.nn import SyncBatchNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def pg_worker(rank, world_size, init_file, func, *args):
    dist.init_process_group(dist.Backend.NCCL, init_method="file://" + init_file, rank=rank, world_size=world_size)
    func(*args)
    dist.destroy_process_group()


def pg_test(world_size=torch.cuda.device_count()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mp.spawn(pg_worker, args=(world_size, tempfile.mkstemp()[1], func, *kwargs.values()), nprocs=world_size)

        globals()["test_" + func.__name__] = wrapper
        return func

    return decorator


def check_parity(torch_bn, fs_bn, x):
    torch_y = torch_bn(x)
    fs_y = fs_bn(x)
    assert torch.allclose(torch_y, fs_y)
    assert torch.allclose(torch_bn.running_mean, fs_bn.running_mean)
    assert torch.allclose(torch_bn.running_var, fs_bn.running_var)


def check_parity_ddp(torch_bn, fs_bn, x):
    yh = torch.ones_like(x)
    rank = dist.get_rank()
    torch_ddp = DDP(torch_bn, device_ids=[rank])
    fs_ddp = DDP(fs_bn, device_ids=[rank])
    torch_bn = torch_ddp.module
    fs_bn = fs_ddp.module
    torch_y = torch_ddp(x)
    fs_y = fs_ddp(x)
    torch_y.backward(yh)
    fs_y.backward(yh)
    assert torch.allclose(torch_y, fs_y)
    assert torch.allclose(torch_bn.running_mean, fs_bn.running_mean)
    assert torch.allclose(torch_bn.running_var, fs_bn.running_var)
    assert torch.allclose(torch_bn.weight.grad, fs_bn.weight.grad), f"{torch_bn.weight.grad} != {fs_bn.weight.grad}"
    assert torch.allclose(torch_bn.bias.grad, fs_bn.bias.grad)


@pg_test(world_size=1)
def parity3d_bn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4, 4).cuda()
    torch_bn = torch.nn.BatchNorm3d(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity(torch_bn, fs_bn, x)


@pg_test()
def parity3d_syncbn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4, 4).cuda() * rank
    torch_bn = torch.nn.SyncBatchNorm(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity_ddp(torch_bn, fs_bn, x)


@pg_test(world_size=1)
def parity2d_bn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4).cuda()
    torch_bn = torch.nn.BatchNorm2d(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity(torch_bn, fs_bn, x)


@pg_test()
def parity2d_syncbn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4).cuda() * rank
    torch_bn = torch.nn.SyncBatchNorm(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity_ddp(torch_bn, fs_bn, x)


@pg_test(world_size=1)
def parity1d_bn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4).cuda()
    torch_bn = torch.nn.BatchNorm1d(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity(torch_bn, fs_bn, x)


@pg_test()
def parity1d_syncbn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4).cuda()
    torch_bn = torch.nn.SyncBatchNorm(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity_ddp(torch_bn, fs_bn, x)
