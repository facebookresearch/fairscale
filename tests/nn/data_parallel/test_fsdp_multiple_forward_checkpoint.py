# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with multiple forward pass + checkpoint. """

import contextlib
import pickle

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from fairscale.fair_dev.testing.testing import dist_init, skip_if_single_gpu, teardown, temp_files_ctx
from fairscale.internal import torch_version
from fairscale.nn import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import auto_wrap_bn
from fairscale.nn.wrap import enable_wrap, wrap


class Model(nn.Module):
    """Model to test FSDP(checkpoint())."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.head(self.block2(self.block1(x)))
        elif isinstance(x, list):
            ys = [self.head(self.block2(self.block1(e))) for e in x]
            return torch.cat(ys, dim=0)


class Model2(nn.Module):
    """Model to test FSDP(checkpoint(), checkpoint())."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=False))
        self.block3 = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(8, 10))

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.head(self.block3(self.block2(self.block1(x))))
        elif isinstance(x, list):
            ys = [self.head(self.block3(self.block2(self.block1(e)))) for e in x]
            return torch.cat(ys, dim=0)


def _create_model(
    with_model2,
    with_sync_bn,
    with_fsdp,
    with_checkpoint,
    mixed_precision,
    flatten,
    wrap_bn,
    fp32_reduce_scatter,
    bucket_cap_mb,
):
    model = Model2() if with_model2 else Model()
    fsdp_config = None
    if with_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        fsdp_config = {
            "mixed_precision": False,
            "flatten_parameters": False,
            "reshard_after_forward": False,
            "bucket_cap_mb": 0,
            "force_input_to_fp32": True,  # SyncBN needs this.
        }

    if with_fsdp and wrap_bn:
        model.block1 = auto_wrap_bn(model.block1, single_rank_pg=False, fsdp_config=fsdp_config)
        model.block2 = auto_wrap_bn(model.block2, single_rank_pg=False, fsdp_config=fsdp_config)
        if with_model2:
            model.block3 = auto_wrap_bn(model.block3, single_rank_pg=False, fsdp_config=fsdp_config)

    if with_checkpoint:
        model.block2 = checkpoint_wrapper(model.block2)
        if with_model2:
            model.block3 = checkpoint_wrapper(model.block3)

    if with_fsdp:
        with enable_wrap(
            wrapper_cls=FSDP,
            flatten_parameters=flatten,
            mixed_precision=mixed_precision,
            compute_dtype=torch.float32,
            fp32_reduce_scatter=fp32_reduce_scatter,
            bucket_cap_mb=bucket_cap_mb,
        ):
            model.block1 = wrap(model.block1)
            model.block2 = wrap(model.block2)
            if with_model2:
                model.block3 = wrap(model.block3)
            model.head = wrap(model.head)

    return model


def _distributed_worker(
    gpu_id,
    world_size,
    with_model2,
    with_sync_bn,
    with_fsdp,
    with_checkpoint,
    files,
    mixed_precision,
    flatten,
    wrap_bn,
    fp32_reduce_scatter,
    bucket_cap_mb,
):
    filename, filename_rpc = files[:2]
    filename_loss = files[2:]

    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, filename, filename_rpc)
    assert result, "Dist init failed"

    # use False below to debug since error msg is not as good with cudnn.
    torch.backends.cudnn.enabled = True

    # these make things deterministic.
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure we have multiple forward passes.
    batch = [
        torch.randn(size=(2, 3, 16, 16)).cuda(),
        torch.randn(size=(2, 3, 9, 9)).cuda(),
        torch.randn(size=(2, 3, 9, 9)).cuda(),
    ]

    if mixed_precision and not with_fsdp:
        batch = [x.half() for x in batch]

    model = _create_model(
        with_model2,
        with_sync_bn,
        with_fsdp,
        with_checkpoint,
        mixed_precision,
        flatten,
        wrap_bn,
        fp32_reduce_scatter,
        bucket_cap_mb,
    )
    model = model.cuda()

    if with_fsdp:
        model = FSDP(
            model,
            flatten_parameters=flatten,
            mixed_precision=mixed_precision,
            compute_dtype=torch.float32,
            fp32_reduce_scatter=fp32_reduce_scatter,
            bucket_cap_mb=bucket_cap_mb,
        )
        model.set_gradient_divide_factors(1.0, 2.0, True)
        no_sync_context = contextlib.suppress()
    else:
        # With DDP, we need no_sync and manual gradient reduction below because
        # it can't handle multiple forward pass + checkpointing otherwise.
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        no_sync_context = model.no_sync()

    mp_context = contextlib.suppress()
    if mixed_precision:
        mp_context = torch.cuda.amp.autocast(enabled=True)

    if gpu_id == 0:
        print(model)

    target = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    losses = {}
    i = 0
    with no_sync_context:
        for iteration in range(3):
            with mp_context:
                out = model(batch)
                loss = criterion(out, target)
                print("Loss", iteration, ":", loss.item())
                losses[f"iter_{i}"] = loss
                i += 1
                optimizer.zero_grad()
                loss.backward()
            # Manual grad reduction, no autocast.
            if not with_fsdp:
                for p in model.parameters():
                    dist.all_reduce(p.grad.data)
                    p.grad.data.div_(2.0)
            # Stepping, no autocast
            optimizer.step()

    # Due to dist.all_reduce code block above with ddp.no_sync, we seem to hit a bug
    # in DDP where tensor.cpu() and torch.save() calls both hang. FSDP is not affected.
    # Therefore, we have to compare losses here instead of states.
    with open(filename_loss[rank], "wb") as f:
        pickle.dump(losses, f)

    teardown()


_result_cache = {}


def _get_cached_results(
    world_size,
    with_model2,
    with_sync_bn,
    with_fsdp,
    with_checkpoint,
    mixed_precision,
    flatten,
    wrap_bn,
    fp32_reduce_scatter,
    bucket_cap_mb,
):
    """Cache the training to save time. For DDP, flatten, wrap_bn etc. doesn't matter, so
    the results can be cached.
    """
    if not with_fsdp:
        flatten = None
        wrap_bn = None
        fp32_reduce_scatter = None

    key = (
        world_size,
        with_model2,
        with_sync_bn,
        with_fsdp,
        with_checkpoint,
        mixed_precision,
        flatten,
        wrap_bn,
        fp32_reduce_scatter,
        bucket_cap_mb,
    )
    global _result_cache
    if key not in _result_cache:
        # Get 4 files: 2 for dist_init and 2 for each rank to save the losses.
        with temp_files_ctx(num=2 + world_size) as temp_files:
            mp.spawn(
                _distributed_worker,
                (
                    world_size,
                    with_model2,
                    with_sync_bn,
                    with_fsdp,
                    with_checkpoint,
                    temp_files,
                    mixed_precision,
                    flatten,
                    wrap_bn,
                    fp32_reduce_scatter,
                    bucket_cap_mb,
                ),
                nprocs=world_size,
            )
            final_losses = {}
            for rank in range(world_size):
                with open(temp_files[2 + rank], "rb") as f:
                    for iter_key, loss in pickle.load(f).items():
                        final_losses[f"rank_{rank}_{iter_key}"] = loss
            _result_cache[key] = final_losses
    return _result_cache[key]


@skip_if_single_gpu
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
@pytest.mark.parametrize("wrap_bn", ["auto_wrap_bn", "no_auto_wrap_bn"])
@pytest.mark.parametrize("model_type", ["model1", "model2"])
@pytest.mark.parametrize("bn_type", ["bn", "sync_bn"])
@pytest.mark.skipif(
    torch_version() >= (1, 14, 0),
    reason="Tests broke in Pytorch pre-release version 1.14",
)
def test_multiple_forward_checkpoint(precision, flatten, wrap_bn, model_type, bn_type):
    mixed_precision = precision == "mixed"
    flatten = flatten == "flatten"
    wrap_bn = wrap_bn == "auto_wrap_bn"
    fp32_reduce_scatter = True if mixed_precision else None
    with_model2 = model_type == "model2"
    with_sync_bn = bn_type == "sync_bn"

    if torch_version() >= (1, 7, 0) and torch_version() < (1, 8, 0) and with_sync_bn:
        # SyncBN is buggy in 1.7, errors like:
        # E         File "/home/circleci/venv/lib/python3.8/site-packages/torch/nn/modules/_functions.py", line 13, in forward
        # E           dtype=running_mean.dtype,
        # E       AttributeError: 'NoneType' object has no attribute 'dtype'
        pytest.skip("SyncBatchNorm in 1.7 is buggy")

    if with_sync_bn and not wrap_bn:
        pytest.skip("SyncBatchNorm requires auto_wrap_bn")

    if torch_version() < (1, 8, 0) and flatten:
        # 1.6 and 1.7 throws this error:
        #   RuntimeError: Trying to backward through the graph a second time, but the saved
        #   intermediate results have already been freed. Specify retain_graph=True when calling
        #   backward the first time.
        pytest.skip("older pytorch throws error when flatten is used")

    world_size = 2
    expected_losses = None

    # Ensure ddp == fsdp when modules are called multiple times per forward pass with/without checkpointing, forward
    # counters and reducer bucketing.
    #
    # The bucketing check exists because the asynchronous gradient reduction it induces can interact with multiple
    # forward passes in complex ways. For example, in the midst of a sharded backward pass, `parameter.grad` may only be
    # `None` or an unsharded gradient tensor. The sharded tensor is then set at the end of the backwards pass. But a
    # unit test with bucketing enabled might not catch violations of this invariant. For very small models, like the
    # kind used in this unit test, bucketing will delay gradient reduction until after all the gradient computation is
    # done. If the reduction incorrectly sets `.grad` to the _sharded_ variant, the test might not fail, since the
    # gradient computations have already happened. Toggling bucketing helps verify that gradient reduction and
    # computation interact correctly.
    combinations = []
    for with_fsdp in [False, True]:
        for with_checkpoint in [False, True]:
            if not with_fsdp and with_checkpoint:
                continue
            for with_bucketing in [False, True]:
                if not with_fsdp and with_bucketing:
                    continue
                combinations.append((with_fsdp, with_checkpoint, with_bucketing))
    print("")
    print("Testing the following configurations:")
    for with_fsdp, with_checkpoint, with_bucketing in combinations:
        print(f"  fsdp {with_fsdp} ckpt {with_checkpoint} bucketing {with_bucketing}")

    for with_fsdp, with_checkpoint, with_bucketing in combinations:
        if with_bucketing:
            bucket_cap_mb = 25
        else:
            bucket_cap_mb = 0
        final_losses = _get_cached_results(
            world_size,
            with_model2,
            with_sync_bn,
            with_fsdp,
            with_checkpoint,
            mixed_precision,
            flatten,
            wrap_bn,
            fp32_reduce_scatter,
            bucket_cap_mb,
        )
        if expected_losses is None:
            expected_losses = final_losses
        else:
            print(f"checking: fsdp {with_fsdp} ckpt {with_checkpoint} bucketing {with_bucketing} with ddp+no_ckpt")

            def check(exp, res):
                assert list(exp.keys()) == list(res.keys()), f"{list(exp.keys())} vs. {list(res.keys())}"
                rtol = 1e-4
                atol = 1e-5
                if with_model2 and mixed_precision and torch_version() >= (1, 9, 0):
                    # On CI, with longer model2, mixed precsion and 1.9, even ddp vs. ddp+ckpt has
                    # larger errors.
                    rtol = 1e-3
                    atol = 1e-4
                for key in exp.keys():
                    exp_loss = exp[key]
                    res_loss = res[key]
                    torch.testing.assert_allclose(exp_loss, res_loss, rtol=rtol, atol=atol)

            check(expected_losses, final_losses)
