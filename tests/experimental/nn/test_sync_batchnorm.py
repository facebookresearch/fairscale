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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.experimental.nn import SyncBatchNorm
from fairscale.nn.checkpoint import checkpoint_wrapper

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
    yh = torch.randn_like(x)
    torch_x = x.detach()
    torch_x.requires_grad = True
    torch_y = torch_bn(torch_x)
    torch_y.backward(yh)
    fs_x = x.detach()
    fs_x.requires_grad = True
    fs_y = fs_bn(fs_x)
    fs_y.backward(yh)
    torch.testing.assert_allclose(torch_y, fs_y)
    torch.testing.assert_allclose(torch_bn.running_mean, fs_bn.running_mean)
    torch.testing.assert_allclose(torch_bn.running_var, fs_bn.running_var)
    torch.testing.assert_allclose(torch_bn.weight, fs_bn.weight)
    torch.testing.assert_allclose(torch_bn.bias, fs_bn.bias)
    torch.testing.assert_allclose(torch_bn.weight.grad, fs_bn.weight.grad)
    torch.testing.assert_allclose(torch_bn.bias.grad, fs_bn.bias.grad)
    torch.testing.assert_allclose(torch_x.grad, fs_x.grad)


def check_parity_ddp(torch_bn, fs_bn, x):
    yh = torch.randn_like(x)
    rank = dist.get_rank()
    torch_ddp = DDP(torch_bn, device_ids=[rank])
    torch_bn = torch_ddp.module
    torch_x = x.detach()
    torch_x.requires_grad = True
    torch_y = torch_ddp(torch_x)
    torch_y.backward(yh)
    fs_ddp = DDP(fs_bn, device_ids=[rank])
    fs_bn = fs_ddp.module
    fs_x = x.detach()
    fs_x.requires_grad = True
    fs_y = fs_ddp(fs_x)
    fs_y.backward(yh)
    torch.testing.assert_allclose(torch_y, fs_y)
    torch.testing.assert_allclose(torch_x.grad, fs_x.grad)
    if isinstance(torch_bn, nn.Sequential):
        torch_bn = torch_bn[0]
        fs_bn = fs_bn[0]
    torch.testing.assert_allclose(torch_bn.running_mean, fs_bn.running_mean)
    torch.testing.assert_allclose(torch_bn.running_var, fs_bn.running_var)
    torch.testing.assert_allclose(torch_bn.weight, fs_bn.weight)
    torch.testing.assert_allclose(torch_bn.bias, fs_bn.bias)
    torch.testing.assert_allclose(torch_bn.weight.grad, fs_bn.weight.grad)
    torch.testing.assert_allclose(torch_bn.bias.grad, fs_bn.bias.grad)


@pg_test(world_size=1)
def parity3d_bn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4, 4).cuda()
    torch_bn = torch.nn.BatchNorm3d(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    check_parity(torch_bn, fs_bn, x)


@pytest.mark.skip("broken at head")
def test_parity3d_checkpoint_syncbn():
    assert 1 == 2


# @pg_test()
def parity3d_checkpoint_syncbn():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4, 4).cuda() * rank
    torch_bn = torch.nn.SyncBatchNorm(3).cuda()
    fs_bn = SyncBatchNorm(3).cuda()
    fs_bn = checkpoint_wrapper(fs_bn)
    check_parity_ddp(torch_bn, fs_bn, x)


@pytest.mark.skip("broken at head")
def test_parity3d_checkpoint_syncbn_twice():
    assert 1 == 2


# @pg_test()
def parity3d_checkpoint_syncbn_twice():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)

    x = torch.randn(4, 3, 4, 4, 4).cuda() * rank
    torch_bn = torch.nn.SyncBatchNorm(3)
    torch_bn = nn.Sequential(torch_bn, torch_bn).cuda()
    fs_bn = SyncBatchNorm(3)
    fs_bn = nn.Sequential(fs_bn, fs_bn).cuda()
    fs_bn = checkpoint_wrapper(fs_bn)
    check_parity_ddp(torch_bn, fs_bn, x)


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


@pg_test()
def memory_allocated():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    x = torch.randn(50, 2048, 7, 7).to(rank)
    torch_bn = torch.nn.SyncBatchNorm(2048).cuda()
    torch_bn = DDP(torch_bn, device_ids=[rank])
    fs_bn = SyncBatchNorm(2048).cuda()
    fs_bn = DDP(fs_bn, device_ids=[rank])
    torch_x = x.detach()
    torch_x.requires_grad = True
    fs_x = x.detach()
    fs_x.requires_grad = True
    torch.cuda.empty_cache()
    mem_at_start = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    torch_y = torch_bn(torch_x)
    torch.cuda.empty_cache()
    mem_after_torch = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    fs_y = fs_bn(fs_x)
    torch.cuda.empty_cache()
    mem_final = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    torch_used = mem_after_torch - mem_at_start
    fs_used = mem_final - mem_after_torch
    assert fs_used < (torch_used * 1.01), f"{fs_used} < {torch_used * 1.01}"
