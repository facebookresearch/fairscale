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

    def __init__(self, params_in_root):
        super().__init__()
        self.params_in_root = params_in_root
        self.child1 = Linear(3, 3, bias=False)
        self.child2 = Linear(3, 3, bias=False)
        if self.params_in_root:
            self.layer_in_root_fsdp = Linear(3, 3, bias=False)

    def forward(self, x):
        # Same layers, once.
        x = self.child1(x)
        x = self.child2(x)
        # Twice.
        x = self.child1(x)
        x = self.child2(x)
        # Optionally, params in root FSDP.
        if self.params_in_root:
            x = self.layer_in_root_fsdp(x)
        return x


def _test_func(rank, world_size, model, fsdp_config, checkpoint, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    model.cuda()
    assert isinstance(fsdp_config, dict), str(fsdp_config)
    assert checkpoint in [True, False], checkpoint

    # Inner FSDPs.
    model.child1 = FSDP(model.child1, **fsdp_config)
    model.child2 = FSDP(model.child2, **fsdp_config)
    # Optionally checkpoint them.
    if checkpoint:
        model.child1 = checkpoint_wrapper(model.child1)
        model.child2 = checkpoint_wrapper(model.child2)
    # Root FSDP.
    model = FSDP(model, **fsdp_config)
    # Use Adam to increase coverage
    optim = Adam(model.parameters(), lr=5e-4)

    for _ in range(5):
        in_data = torch.rand(64, 3).cuda()
        in_data.requires_grad = True
        out = model(in_data)
        # without double fwd, no_params_in_root assert on backward
        # since we have multiple checkpointed blocks.
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
@pytest.mark.parametrize("params_in_root", [True, False])
def test_it(fsdp_config, checkpoint, params_in_root):
    """Test FSDP with uneven divide of parameter shards."""
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    world_size = 2

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    model = MyModel(params_in_root)

    mp.spawn(
        _test_func,
        args=(world_size, model, fsdp_config, checkpoint, temp_file_name, unused),
        nprocs=world_size,
        join=True,
    )
