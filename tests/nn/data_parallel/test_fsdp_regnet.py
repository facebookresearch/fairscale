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
from torch.nn import BatchNorm2d, Conv2d, Module, SyncBatchNorm
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


class Model(Module):
    def __init__(self):
        super().__init__()
        # TODO (Min): for now, we just test pytorch sync_bn here.
        #             this will grow into regnet; testing apex sync_bn, etc.
        self.conv = Conv2d(2, 2, (1, 1))
        self.bn = BatchNorm2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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
    world_size = 2
    iterations = 100
    inputs = [[]] * world_size
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

    model = Model()
    model.load_state_dict(state_before)
    model = model.cuda()

    if ddp:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    else:
        # Note, different rank may wrap in different order due to different random
        # seeds. But results should be the same.
        if random.randint(0, 1) == 0:
            print("auto_wrap_bn, then convert_sync_batchnorm")
            model = auto_wrap_bn(model)
            model = SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            print("convert_sync_batchnorm, then auto_wrap_bn")
            model = SyncBatchNorm.convert_sync_batchnorm(model)
            model = auto_wrap_bn(model)
        model = FSDP(model, **fsdp_config).cuda()
    optim = SGD(model.parameters(), lr=0.1)

    for in_data in inputs[rank]:
        in_data = in_data.cuda()
        context = contextlib.suppress()
        if ddp and ddp_mixed_precision:
            in_data = in_data.half()
            context = torch.cuda.amp.autocast(enabled=True)
        with context:
            out = model(in_data)
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

    world_size = 2
    mp.spawn(
        _test_func,
        args=(world_size, fsdp_config, None, temp_files[0], temp_files[1], state_before, inputs, None, state_after),
        nprocs=world_size,
        join=True,
    )
