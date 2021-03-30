# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with regnet-like model. """

import contextlib
import random
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, Module, ReLU, Sequential, Sigmoid, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState, auto_wrap_bn
from fairscale.utils.testing import (
    dist_init,
    objects_are_equal,
    rmf,
    skip_if_single_gpu,
    state_dict_norm,
    teardown,
    torch_version,
)

# Const test params.
#   Reduce iterations to 1 for debugging.
#   Change world_size to 8 on beefy machines for better test coverage.
_world_size = 2
_iterations = 5

# TODO (Min): test inplace relu in the future.
_relu_inplace = False

# TODO (Min): test apex BN when available in the future.
try:
    import apex

    apex_bn_converter = apex.parallel.convert_syncbn_model
except ImportError:
    apex_bn_converter = None
pytorch_bn_converter = SyncBatchNorm.convert_sync_batchnorm  # type: ignore
_bn_converter = pytorch_bn_converter
_single_rank_pg = False


class ResBlock(Module):
    """Conv block in regnet with residual connection."""

    def __init__(self, width_in, width_out):
        super().__init__()
        self.proj = Conv2d(width_in, width_out, (1, 1), (2, 2), bias=False)
        self.bn = BatchNorm2d(width_out)
        self.f = Sequential(
            Sequential(  # block a
                Conv2d(width_in, width_out, (1, 1), (1, 1), bias=False), BatchNorm2d(width_out), ReLU(_relu_inplace),
            ),
            Sequential(  # block b
                Conv2d(width_out, width_out, (3, 3), (2, 2), (1, 1), groups=2, bias=False),
                BatchNorm2d(width_out),
                ReLU(_relu_inplace),
            ),
            Sequential(  # block se
                AdaptiveAvgPool2d((1, 1)),
                Sequential(
                    Conv2d(width_out, 2, (1, 1), (1, 1), bias=False),
                    ReLU(_relu_inplace),
                    Conv2d(2, width_out, (1, 1), (1, 1), bias=False),
                    Sigmoid(),
                ),
            ),
            Conv2d(width_out, width_out, (1, 1), (1, 1), bias=False),  # block c
            BatchNorm2d(width_out),  # final_bn
        )
        self.relu = ReLU()
        self.need_fsdp_wrap = True

    def forward(self, x):
        x = self.bn(self.proj(x)) + self.f(x)
        return self.relu(x)


class Model(Module):
    """SSL model with trunk and head."""

    def __init__(self):
        super().__init__()
        # trunk
        self.trunk = Sequential()
        self.trunk.need_fsdp_wrap = True  # Set a flag for later wrapping.
        stem = Sequential(Conv2d(2, 4, (3, 3), (2, 2), (1, 1), bias=False), BatchNorm2d(4), ReLU(_relu_inplace),)
        any_stage_block1_0 = ResBlock(4, 8)
        self.trunk.add_module("stem", stem)
        self.trunk.add_module("any_stage_block1", Sequential(any_stage_block1_0))

        # head
        self.head = Sequential(
            # TODO (Min): FSDP-mixed_precision doesn't compute the same ways as DDP AMP when bias=True.
            #             so, we use bias=False for now in the projection_head.
            Sequential(Linear(16, 16, bias=False), ReLU(), Linear(16, 8, bias=False),),  # projection_head
            Linear(8, 15, bias=False),  # prototypes0
        )

    def forward(self, x):
        x = self.trunk(x).reshape(-1)
        x = self.head(x)
        return x


# We get a bit fancy here. Since the scope is `module`, this is run only
# once no matter how many tests variations for FSDP are requested to run
# to compare with the DDP reference. For example, a single DDP
# reference run is needed for both flatten and non-flatten param FSDP.
#
# Note, this runs DDP twice with and without mixed precision and asserts
# the resulting weights are different.
#
# This fixture captures and returns:
#
#   - model state_dict before training
#   - model data inputs
#   - model state_dict after training
@pytest.fixture(scope="module")
def ddp_ref():
    # Get a reference model state
    model = Model()
    state_before = model.state_dict()

    # Get reference inputs per rank.
    world_size = _world_size
    iterations = _iterations
    print(f"Getting DDP reference for world_size {world_size} and iterations {iterations}")
    inputs = [[] for i in range(world_size)]
    for rank in range(world_size):
        for i in range(iterations):
            inputs[rank].append(torch.rand(2, 2, 2, 2))

    # Run DDP training twice, fp and mp.
    for precision in ["full", "mixed"]:
        temp_file_name = tempfile.mkstemp()[1]
        unused = tempfile.mkstemp()[1]
        rank_0_output = tempfile.mkstemp()[1]
        try:
            fsdp_config = None  # This means we use DDP in _test_func.
            mp.spawn(
                _test_func,
                args=(
                    world_size,
                    fsdp_config,
                    precision == "mixed",
                    temp_file_name,
                    unused,
                    state_before,
                    inputs,
                    rank_0_output,
                    None,
                ),
                nprocs=world_size,
                join=True,
            )
            if precision == "full":
                state_after_fp = torch.load(rank_0_output)
            else:
                state_after_mp = torch.load(rank_0_output)
        finally:
            rmf(temp_file_name)
            rmf(unused)
            rmf(rank_0_output)

    assert state_dict_norm(state_after_fp) != state_dict_norm(state_after_mp)

    return state_before, inputs, state_after_fp, state_after_mp


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    yield temp_file_name, unused

    # temp files could have been removed, so we use rmf.
    rmf(temp_file_name)
    rmf(unused)


def _test_func(
    rank,
    world_size,
    fsdp_config,
    ddp_mixed_precision,
    tempfile_name,
    unused,
    state_before,
    inputs,
    rank_0_output,
    state_after,
):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    ddp = True
    if fsdp_config:
        ddp = False
        assert isinstance(fsdp_config, dict), str(fsdp_config)
        if fsdp_config["mixed_precision"]:
            # To match DDP in AMP -O1, we need fp32 reduce scatter.
            fsdp_config["fp32_reduce_scatter"] = True

    model = Model()
    model.load_state_dict(state_before)
    model = model.cuda()

    if ddp:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], broadcast_buffers=True)
    else:
        # Note, different rank may wrap in different order due to different random
        # seeds. But results should be the same.
        if random.randint(0, 1) == 0:
            print("auto_wrap_bn, then convert_sync_batchnorm")
            model = auto_wrap_bn(model, _single_rank_pg)
            model = _bn_converter(model)
        else:
            print("convert_sync_batchnorm, then auto_wrap_bn")
            model = _bn_converter(model)
            model = auto_wrap_bn(model, _single_rank_pg)
        model = FSDP(model, **fsdp_config).cuda()
        # Print the model for verification.
        if rank == 0:
            print(model)
    optim = SGD(model.parameters(), lr=0.1)

    for in_data in inputs[rank]:
        in_data = in_data.cuda()
        context = contextlib.suppress()
        if ddp and ddp_mixed_precision:
            in_data = in_data.half()
            context = torch.cuda.amp.autocast(enabled=True)
        if not ddp and fsdp_config["mixed_precision"]:
            context = torch.cuda.amp.autocast(enabled=True)
        with context:
            out = model(in_data)
            # TODO (Min): this loss is causing nan after ~10 iters, need a real loss and grad scaler.
            loss = out.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    if ddp:
        # Save the rank 0 state_dict to the output file.
        if rank == 0:
            state_after = model.module.cpu().state_dict()
            torch.save(state_after, rank_0_output)
    else:
        model.assert_state(TrainingState.IDLE)
        # Ensure final state equals to the state_after.
        fsdp_state = model.state_dict()
        # Move tensors to CPU to compare numerics.
        for k, v in fsdp_state.items():
            fsdp_state[k] = v.cpu()
        # Enable for debugging.
        if False and rank == 0:

            def dump(d):
                for k, v in d.items():
                    print(k, v)

            dump(state_after)
            dump(fsdp_state)
        assert objects_are_equal(state_after, fsdp_state, raise_exception=True)

    teardown()


# We use strings for precision and flatten params instead of bool to
# make the pytest output more readable.
@skip_if_single_gpu
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
def test1(temp_files, ddp_ref, precision, flatten):
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    state_before, inputs, state_after_fp, state_after_mp = ddp_ref

    if precision == "full":
        state_after = state_after_fp
    else:
        state_after = state_after_mp

    fsdp_config = {}
    fsdp_config["mixed_precision"] = precision == "mixed"
    fsdp_config["flatten_parameters"] = flatten == "flatten"

    world_size = _world_size
    mp.spawn(
        _test_func,
        args=(world_size, fsdp_config, None, temp_files[0], temp_files[1], state_before, inputs, None, state_after),
        nprocs=world_size,
        join=True,
    )
