# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with shared weights between wrappers. """

from copy import deepcopy

import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.utils.testing import dist_init, objects_are_equal, skip_if_no_cuda, teardown, temp_files_ctx


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    # dist_init needs 2 files
    with temp_files_ctx(2) as files:
        yield files


@skip_if_no_cuda
@pytest.mark.parametrize("outer_flat", ["outer_flat", "outer_nonflat"])
@pytest.mark.parametrize("inner_flat", ["inner_flat", "inner_nonflat"])
@pytest.mark.parametrize("share_bias", ["share_bias", "no_share_bias"])
@pytest.mark.parametrize("has_bias", ["has_bias", "no_bias"])
def test_shared_weight(temp_files, outer_flat, inner_flat, share_bias, has_bias):
    """Test FSDP with a model with shared weights."""

    outer_flat = outer_flat == "outer_flat"
    inner_flat = inner_flat == "inner_flat"
    share_bias = share_bias == "share_bias"
    has_bias = has_bias == "has_bias"

    result = dist_init(rank=0, world_size=1, filename=temp_files[0], filename_rpc=temp_files[1])
    assert result, "Dist init failed"

    class Model(Module):
        def __init__(self, with_fsdp=False):
            super().__init__()
            self.l1 = Linear(4, 4, bias=has_bias).cuda()
            self.l2 = Linear(4, 4, bias=has_bias).cuda()
            self.l3 = Linear(4, 4, bias=has_bias).cuda()

            # share the weights.
            self.l1.weight = self.l3.weight
            if has_bias and share_bias:
                self.l1.bias = self.l3.bias

            if with_fsdp:
                self.l1 = FSDP(self.l1, flatten_parameters=inner_flat)
                self.l2 = FSDP(self.l2, flatten_parameters=inner_flat)
                self.l3 = FSDP(self.l3, flatten_parameters=inner_flat)

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            return x

    model = Model()
    fsdp_model = FSDP(Model(with_fsdp=True), flatten_parameters=outer_flat)
    sd_before = deepcopy(model.state_dict())
    fsdp_model.load_state_dict(sd_before)
    in_data = torch.rand(1, 4).cuda()

    _train(model, in_data)
    _train(fsdp_model, in_data)

    # Before and after state should not be equal.
    assert not objects_are_equal(sd_before, model.state_dict())

    if not inner_flat:
        # Non FSDP an FSDP should be equal.
        objects_are_equal(model.state_dict(), fsdp_model.state_dict(), raise_exception=True)
    else:
        # Otherwise, weight sharing should cause them to be not equal.
        assert not objects_are_equal(model.state_dict(), fsdp_model.state_dict())

    teardown()


def _train(model, in_data):
    optim = SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        out = model(in_data)
        out.sum().backward()
        optim.step()
        model.zero_grad(set_to_none=True)
