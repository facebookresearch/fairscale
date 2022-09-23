# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with regnet-like model. """

import contextlib
from itertools import product
import random
import tempfile

import pytest
import torch
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    SyncBatchNorm,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

from fairscale.fair_dev.testing.testing import (
    dist_init,
    objects_are_equal,
    rmf,
    skip_if_single_gpu,
    state_dict_norm,
    teardown,
    torch_cuda_version,
)
from fairscale.internal import torch_version
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState, auto_wrap_bn

if torch_version() >= (1, 8, 0):
    from fairscale.optim.grad_scaler import ShardedGradScaler

# Const test params.
#   Reduce iterations to 1 for debugging.
#   Change world_size to 8 on beefy machines for better test coverage.
_world_size = 2
_iterations = 5

# Cover different ReLU flavors. Different workers may have different values since
# this is a file level global. This is intensional to cover different behaviors.
_relu_inplace = True
if random.randint(0, 1) == 0:
    _relu_inplace = False

# TODO (Min): test apex BN when available in the future.
try:
    import apex

    apex_bn_converter = apex.parallel.convert_syncbn_model
except ImportError:
    apex_bn_converter = None
pytorch_bn_converter = SyncBatchNorm.convert_sync_batchnorm  # type: ignore
_single_rank_pg = False


class ResBlock(Module):
    """Conv block in regnet with residual connection."""

    def __init__(self, width_in, width_out):
        super().__init__()
        self.proj = Conv2d(width_in, width_out, (1, 1), (2, 2), bias=False)
        self.bn = BatchNorm2d(width_out)
        self.f = Sequential(
            Sequential(  # block a
                Conv2d(width_in, width_out, (1, 1), (1, 1), bias=False),
                BatchNorm2d(width_out),
                ReLU(_relu_inplace),
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

    def __init__(self, conv_bias, linear_bias):
        super().__init__()
        print(f"relu inplace: {_relu_inplace}, conv bias: {conv_bias}, linear bias: {linear_bias}")

        self.trunk = Sequential()
        self.trunk.need_fsdp_wrap = True  # Set a flag for later wrapping.
        stem = Sequential(Conv2d(2, 4, (3, 3), (2, 2), (1, 1), bias=conv_bias), BatchNorm2d(4), ReLU(_relu_inplace))
        any_stage_block1_0 = ResBlock(4, 8)
        self.trunk.add_module("stem", stem)
        self.trunk.add_module("any_stage_block1", Sequential(any_stage_block1_0))

        self.head = Sequential(
            Sequential(Linear(16, 16, bias=linear_bias), ReLU(), Linear(16, 8, bias=linear_bias)),  # projection_head
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
    # Cover different bias flavors. Use random instead of parameterize them to reduce
    # the test runtime. Otherwise, we would have covered all cases exhaustively.
    conv_bias = True
    if random.randint(0, 1) == 0:
        conv_bias = False
    linear_bias = True
    if random.randint(0, 1) == 0:
        linear_bias = False

    # Get a reference model state
    model = Model(conv_bias, linear_bias)
    state_before = model.state_dict()

    # Get reference inputs per rank.
    world_size = _world_size
    iterations = _iterations
    print(f"Getting DDP reference for world_size {world_size} and iterations {iterations}")
    inputs = [[] for i in range(world_size)]
    for rank in range(world_size):
        for i in range(iterations):
            inputs[rank].append(torch.rand(2, 2, 2, 2))

    # Run reference DDP training 4 times, fp and mp, sync_bn or not.
    state_after = {}
    for precision, sync_bn in product(["full", "mixed"], ["none", "pytorch"]):
        temp_file_name = tempfile.mkstemp()[1]
        unused = tempfile.mkstemp()[1]
        rank_0_output = tempfile.mkstemp()[1]
        try:
            fsdp_config = None  # This means we use DDP in _distributed_worker.
            mp.spawn(
                _distributed_worker,
                args=(
                    world_size,
                    fsdp_config,
                    None,
                    precision == "mixed",
                    temp_file_name,
                    unused,
                    state_before,
                    inputs,
                    rank_0_output,
                    None,
                    sync_bn,
                    conv_bias,
                    linear_bias,
                ),
                nprocs=world_size,
                join=True,
            )
            state_after[(precision, sync_bn)] = torch.load(rank_0_output)
        finally:
            rmf(temp_file_name)
            rmf(unused)
            rmf(rank_0_output)

    # Sanity check DDP's final states.
    states = list(state_after.values())
    for state in states[1:]:
        assert state_dict_norm(states[0]) != state_dict_norm(state)

    return state_before, inputs, conv_bias, linear_bias, state_after


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    yield temp_file_name, unused

    # temp files could have been removed, so we use rmf.
    rmf(temp_file_name)
    rmf(unused)


def _distributed_worker(
    rank,
    world_size,
    fsdp_config,
    fsdp_wrap_bn,
    ddp_mixed_precision,
    tempfile_name,
    unused,
    state_before,
    inputs,
    rank_0_output,
    state_after,
    sync_bn,
    conv_bias,
    linear_bias,
):
    torch.backends.cudnn.deterministic = True

    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    ddp = True
    if fsdp_config:
        ddp = False
        assert isinstance(fsdp_config, dict), str(fsdp_config)
        if fsdp_config["mixed_precision"]:
            # To match DDP in AMP -O1, we need fp32 reduce scatter.
            fsdp_config["fp32_reduce_scatter"] = True

    model = Model(conv_bias, linear_bias)
    model.load_state_dict(state_before)
    model = model.cuda()

    class DummyScaler:
        def scale(self, loss):
            return loss

        def step(self, optim):
            optim.step()

        def update(self):
            pass

    scaler = DummyScaler()
    if ddp:
        if sync_bn == "pytorch":
            model = pytorch_bn_converter(model)
        model = DDP(model, device_ids=[rank], broadcast_buffers=True)
        if ddp_mixed_precision:
            scaler = GradScaler()
    else:
        # Note, different rank may wrap in different order due to different random
        # seeds. But results should be the same.
        if random.randint(0, 1) == 0:
            print(f"auto_wrap_bn {fsdp_wrap_bn}, then sync_bn {sync_bn}")
            if fsdp_wrap_bn:
                model = auto_wrap_bn(model, _single_rank_pg)
            if sync_bn == "pytorch":
                model = pytorch_bn_converter(model)
        else:
            print(f"sync_bn {sync_bn}, then auto_wrap_bn {fsdp_wrap_bn}")
            if sync_bn == "pytorch":
                model = pytorch_bn_converter(model)
            if fsdp_wrap_bn:
                model = auto_wrap_bn(model, _single_rank_pg)
        model = FSDP(model, **fsdp_config).cuda()
        if fsdp_config["mixed_precision"]:
            scaler = ShardedGradScaler()
        # Print the model for verification.
        if rank == 0:
            print(model)
    optim = SGD(model.parameters(), lr=0.1)
    loss_func = CrossEntropyLoss()

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
            fake_label = torch.zeros(1, dtype=torch.long).cuda()
            loss = loss_func(out.unsqueeze(0), fake_label)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
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
        # Change False to True to enable this when you want to debug the mismatch.
        if False and rank == 0:

            def dump(d):
                for k, v in d.items():
                    print(k, v)

            dump(state_after)
            dump(fsdp_state)
        # If sync_bn is used, all ranks should have the same state, so we can compare with
        # rank 0 state on every rank. Otherwise, only compare rank 0 with rank 0.
        if sync_bn != "none" or rank == 0:
            assert objects_are_equal(state_after, fsdp_state, raise_exception=True)

    teardown()


# We use strings for precision and flatten params instead of bool to
# make the pytest output more readable.
@pytest.mark.skip("broken at head")
@skip_if_single_gpu
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
@pytest.mark.parametrize("sync_bn", ["none", "pytorch"])
def test_regnet(temp_files, ddp_ref, precision, flatten, sync_bn):
    if torch_version() < (1, 8, 0):
        pytest.skip("pytorch version >= 1.8.0 required")

    state_before, inputs, conv_bias, linear_bias, state_after = ddp_ref

    state_after = state_after[(precision, sync_bn)]

    fsdp_config = {}
    fsdp_config["mixed_precision"] = precision == "mixed"
    fsdp_config["flatten_parameters"] = flatten == "flatten"

    # When linear bias is True, DDP's AMP O1 and FSDP's default AMP O1.5 is different,
    # we force FSDP to use AMP O1 here by setting compute_dtype to float32.
    if linear_bias:
        fsdp_config["compute_dtype"] = torch.float32

    if fsdp_config["mixed_precision"] and torch_cuda_version() < (11, 0):
        pytest.skip("Only CUDA 11 is supported with AMP equivalency")

    # Wrap BN half of the time.
    wrap_bn = True
    if random.randint(0, 1) == 0:
        wrap_bn = False
    # Except, always wrap BN in mixed precision + sync_bn mode, due to error of sync_bn wrapping,
    # regardless of compute_dtype.
    if fsdp_config["mixed_precision"] and sync_bn != "none":
        wrap_bn = True

    # When BN is not wrapped (i.e. not in full precision), FSDP's compute_dtype needs to
    # be fp32 to match DDP (otherwise, numerical errors happen on BN's running_mean/running_var
    # buffers).
    if fsdp_config["mixed_precision"] and not wrap_bn:
        fsdp_config["compute_dtype"] = torch.float32

    world_size = _world_size
    mp.spawn(
        _distributed_worker,
        args=(
            world_size,
            fsdp_config,
            wrap_bn,
            None,
            temp_files[0],
            temp_files[1],
            state_before,
            inputs,
            None,
            state_after,
            sync_bn,
            conv_bias,
            linear_bias,
        ),
        nprocs=world_size,
        join=True,
    )
