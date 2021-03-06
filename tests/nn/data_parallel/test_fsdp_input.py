# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with different input types. """

import os
import random

import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState
from fairscale.utils.testing import skip_if_no_cuda, torch_version


# We only test on GPU since mix-precision only really works on GPU.
@skip_if_no_cuda
@pytest.mark.parametrize(
    "fsdp_config", [{}, {"mixed_precision": True}],
)
@pytest.mark.parametrize("input_cls", [dict, list])
def test_it(fsdp_config, input_cls):
    """Test FSDP with input being a list or a dict, only single GPU."""
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    # Random port in case the next test run quickly, same port would cause conflict.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(2000, 3000))
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    try:
        assert isinstance(fsdp_config, dict), str(fsdp_config)

        class Model(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(4, 4)

            def forward(self, input):
                if isinstance(input, list):
                    input = input[0]
                else:
                    assert isinstance(input, dict), input
                    input = input["in"]
                return self.layer(input)

        model = FSDP(Model(), **fsdp_config).cuda()
        optim = SGD(model.parameters(), lr=0.1)

        for _ in range(5):
            in_data = torch.rand(64, 4).cuda()
            in_data.requires_grad = True
            if input_cls is list:
                in_data = [in_data]
            else:
                assert input_cls is dict
                in_data = {"in": in_data}

            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        model.assert_state(TrainingState.IDLE)

    finally:
        # Clean-up is important or the next test in this file may fail to init the PG.
        torch.distributed.destroy_process_group()
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]
