# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with nested wrapping multiple times. """

import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState
from fairscale.utils import torch_version
from fairscale.utils.testing import dist_init, skip_if_no_cuda, teardown


def _test_func(rank, world_size, fsdp_config, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)

    class InnerModel(Module):
        def __init__(self):
            super().__init__()
            self.layers = Sequential(FSDP(Linear(5, 5), **fsdp_config),)

        def forward(self, x):
            return self.layers(x)

    inner_model = InnerModel()
    model = FSDP(inner_model, **fsdp_config).cuda()
    optim = SGD(model.parameters(), lr=0.1)

    for i in range(3):
        input = torch.rand((1, 5), dtype=torch.float).cuda()
        input.requires_grad = True
        output = model(input)
        output.sum().backward()
        optim.step()
        optim.zero_grad()
    input = torch.rand((1, 5), dtype=torch.float).cuda()
    output = model(input)

    model.assert_state(TrainingState.IDLE)

    # second time to rewrap the inner model
    rewrapped_model = FSDP(inner_model, **fsdp_config).cuda()
    rewrapped_output = rewrapped_model(input)

    assert torch.allclose(output, rewrapped_output)
    teardown()


# We use strings for precision and flatten instead of bool to
# make the pytest output more readable.
@skip_if_no_cuda
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
def test(world_size, precision, flatten):
    """
    This test simulates wrapping the module after training to run inference.
    This is required in cases where later in a session, the model is wrapped again in FSDP but
    contains nested FSDP wrappers within the module.
    """
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    fsdp_config = {
        "mixed_precision": precision == "mixed",
        "flatten_parameters": flatten == "flatten",
    }

    mp.spawn(
        _test_func, args=(world_size, fsdp_config, temp_file_name, unused), nprocs=world_size, join=True,
    )
