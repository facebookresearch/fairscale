# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with forward pass with multiple calls of a module, like a loop. """

import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import Linear, Module
from torch.optim import Adam

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel.fully_sharded_data_parallel import TrainingState
from fairscale.nn.misc import checkpoint_wrapper
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, torch_version


class MyModel(Module):
    """Model that forward multiple times through the same module."""

    def __init__(self):
        super().__init__()
        self.layer = Linear(3, 3, bias=False)

    def forward(self, x):
        # Same layer, multiple times.
        x = self.layer(x)
        # x = self.layer(x)
        return x


def _test_func(rank, world_size, model, fsdp_config, checkpoint, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    model.cuda()
    assert isinstance(fsdp_config, dict), str(fsdp_config)
    assert checkpoint in [True, False], checkpoint

    model.layer = FSDP(model.layer, **fsdp_config)
    if checkpoint:
        model.layer = checkpoint_wrapper(model.layer)
    model = FSDP(model, **fsdp_config)
    optim = Adam(model.parameters(), lr=5e-4)

    for _ in range(5):
        in_data = torch.rand(64, 3).cuda()
        in_data.requires_grad = True
        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    model.assert_state(TrainingState.IDLE)
    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize(
    "fsdp_config", [{}, {"flatten_parameters": False}],
)
@pytest.mark.parametrize("checkpoint", [True, False])
def test_it(fsdp_config, checkpoint):
    """Test FSDP with uneven divide of parameter shards."""
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    world_size = 2

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    model = MyModel()

    mp.spawn(
        _test_func,
        args=(world_size, model, fsdp_config, checkpoint, temp_file_name, unused),
        nprocs=world_size,
        join=True,
    )
