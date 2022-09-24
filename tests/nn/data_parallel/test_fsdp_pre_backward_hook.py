# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with pre-backward hook bug. """

import pytest
import torch
from torch.nn import Linear, Module

from fairscale.fair_dev.testing.testing import dist_init, skip_if_no_cuda, teardown, temp_files_ctx
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    # dist_init needs 2 files
    with temp_files_ctx(2) as files:
        yield files


@skip_if_no_cuda
def test_pre_backward_hook(temp_files):
    """Test FSDP with a model that triggers a pre_backward hook bug."""

    result = dist_init(rank=0, world_size=1, filename=temp_files[0], filename_rpc=temp_files[1])
    assert result, "Dist init failed"

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(4, 4).cuda()
            self.l2 = FSDP(Linear(4, 4).cuda())
            self.l3 = Linear(4, 4).cuda()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            inner_result = x
            x = self.l3(x)
            return x, inner_result

        def assert_and_clear_grad(self):
            for p in self.parameters():
                assert p.shape in [(4, 4), (4,), (4 * 4 + 4,)], p.shape
                assert p.grad is not None
                p.grad = None

    model = FSDP(Model(), flatten_parameters=False).cuda()
    in_data = torch.rand(1, 4).cuda()
    for _ in range(3):
        out, _ = model(in_data)
        out.sum().backward()
        model.assert_and_clear_grad()

    teardown()
