# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with regnet-like model. """

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import BatchNorm2d, Conv2d, Module, SyncBatchNorm
from torch.optim import SGD

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, torch_version


def _test_func(rank, world_size, fsdp_config, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)

    class Model(Module):
        def __init__(self):
            super().__init__()
            # TODO (Min): for now, we just test pytorch sync_bn here.
            #             this will grow into regnet; testing apex sync_bn, etc.
            self.conv = Conv2d(2, 2, (1, 1))
            # Put BN in is own FP32, unflatten, single GPU group FSDP.
            # Note, SyncBNs still have a group size == world_size.
            # The input and output for BN are still FP16. See ``keep_batchnorm_fp32``
            # here: https://nvidia.github.io/apex/amp.html
            self.bn = FSDP(
                BatchNorm2d(2),
                mixed_precision=False,
                process_group=dist.new_group(ranks=[rank]),
                flatten_parameters=False,
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return x

    # TODO (Min): check DDP equivalency.

    model = Model()
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = FSDP(model, **fsdp_config).cuda()
    optim = SGD(model.parameters(), lr=0.1)

    for _ in range(3):
        in_data = torch.rand(2, 2, 2, 2).cuda()
        in_data.requires_grad = True
        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    model.assert_state(TrainingState.IDLE)
    teardown()


# We use strings for precision and flatten instead of bool to
# make the pytest output more readable.
@skip_if_single_gpu
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
def test1(precision, flatten):
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    fsdp_config = {}
    fsdp_config["mixed_precision"] = precision == "mixed"
    fsdp_config["flatten_parameters"] = flatten == "flatten"

    # Some bugs only show up when we are in world_size > 1 due to sharding changing
    # the tensor dimensions.
    world_size = 2
    mp.spawn(
        _test_func, args=(world_size, fsdp_config, temp_file_name, unused), nprocs=world_size, join=True,
    )
