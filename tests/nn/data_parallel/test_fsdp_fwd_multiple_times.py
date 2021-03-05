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
from torch.nn import Linear, Module, Sequential
from torch.optim import Adam

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel.fully_sharded_data_parallel import TrainingState
from fairscale.nn.misc import checkpoint_wrapper as ckpt
from fairscale.nn.wrap import enable_wrap, wrap
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, torch_version

# To make it simpler to write and easier to understand, we do
# FSDP and checkpointing in the Model classes explicitly.
#
# This means the models are created at the workers not by the
# controller/main process.
#
# For this test, we don't do any numerical checking, so the weights
# of the models at the each worker doesn't matter and we don't make
# them the same.


class ModelCkpt1Level(Module):
    """ Model that has one level checkpoint."""

    def __init__(self):
        super().__init__()
        self.child1 = ckpt(Linear(3, 3))
        self.child2 = ckpt(Linear(3, 3))

    def forward(self, x):
        x = self.child1(x)
        x = self.child2(x)
        return x


class ModelCkpt2Level(Module):
    """ Model that has two level checkpoint."""

    def __init__(self):
        super().__init__()
        self.child1 = ckpt(Sequential(ckpt(Linear(3, 3)), ckpt(Linear(3, 3))))
        self.child2 = ckpt(Sequential(ckpt(Linear(3, 3)), ckpt(Linear(3, 3))))

    def forward(self, x):
        x = self.child1(x)
        x = self.child2(x)
        return x


class ModelFSDP1Level(Module):
    """ Model that has one level checkpoint."""

    def __init__(self):
        super().__init__()
        self.child1 = wrap(Linear(3, 3))
        self.child2 = wrap(Linear(3, 3))

    def forward(self, x):
        x = self.child1(x)
        x = self.child2(x)
        return x


class ModelFSDPCkpt2Level(Module):
    """ Model that has two level checkpoint."""

    def __init__(self):
        super().__init__()
        self.child1 = ckpt(wrap(Sequential(wrap(Linear(3, 3)), wrap(Linear(3, 3)))))
        self.child2 = ckpt(wrap(Sequential(wrap(Linear(3, 3)), wrap(Linear(3, 3)))))
        self.root = Linear(3, 3)

    def forward(self, x):
        x = self.child1(x)
        x = self.child2(x)
        return x


# List of all Models.
MODELS = [ModelCkpt1Level, ModelCkpt2Level, ModelFSDP1Level, ModelFSDPCkpt2Level]


def _test_func(rank, world_size, model_cls, fsdp_config, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)
    fsdp_config["wrapper_cls"] = FSDP
    with enable_wrap(**fsdp_config):
        model = wrap(model_cls()).cuda()

    # Use Adam to increase overall coverage among all FSDP tests.
    optim = Adam(model.parameters(), lr=5e-4)

    for _ in range(5):
        in_data = torch.rand(64, 3).cuda()
        in_data.requires_grad = True
        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    if hasattr(model, "assert_state"):
        model.assert_state(TrainingState.IDLE)
    teardown()


@skip_if_single_gpu
@pytest.mark.parametrize(
    # Sweep both flatten and not flatten since it likely matters here.
    "fsdp_config",
    [{}, {"flatten_parameters": False}],
)
@pytest.mark.parametrize("model_cls", MODELS)
def test_it(fsdp_config, model_cls):
    """Test FSDP and checkpoint_wrapper with different model configs."""
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    world_size = 2

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    mp.spawn(
        _test_func, args=(world_size, model_cls, fsdp_config, temp_file_name, unused), nprocs=world_size, join=True,
    )
