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
import torch.multiprocessing as mp
from torch.nn import Linear, Module
from torch.optim import SGD

from fairscale.fair_dev.testing.testing import (
    dist_init,
    objects_are_equal,
    skip_if_single_gpu,
    teardown,
    temp_files_ctx,
)
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


class Model(Module):
    def __init__(self, with_fsdp=False, inner_flat=False, sharing=None):
        super().__init__()
        self.l0 = Linear(4, 4, bias=True).cuda()
        self.l1 = Linear(4, 4, bias=True).cuda()
        self.l2 = Linear(4, 4, bias=True).cuda()
        self.l3 = Linear(4, 4, bias=True).cuda()

        # share the weights. the layer must have at least 1 param is that's not
        # shared. Therefore, we have bias=True and testing either sharing the
        # weight or the bias.
        if sharing == "share_only_weights":
            self.l1.weight = self.l3.weight
        elif sharing == "share_only_bias":
            self.l1.bias = self.l3.bias
        else:
            assert sharing is None or sharing == "share_none"

        if with_fsdp:
            # Shared layers much be un-flatten.
            self.l1 = FSDP(self.l1, flatten_parameters=False)
            self.l2 = FSDP(self.l2, flatten_parameters=inner_flat)
            self.l3 = FSDP(self.l3, flatten_parameters=False)

            if sharing in ["share_only_weights"]:
                self.l3.append_shared_param(self.l1.module.weight)
            if sharing in ["share_only_bias"]:
                self.l3.append_shared_param(self.l1.module.bias)

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    # dist_init needs 2 files + 3 files for before state, after state, in_data.
    with temp_files_ctx(5) as files:
        yield files


@skip_if_single_gpu
@pytest.mark.parametrize("outer_flat", ["outer_flat", "outer_nonflat"])
@pytest.mark.parametrize("inner_flat", ["inner_flat", "inner_nonflat"])
@pytest.mark.parametrize("sharing", ["share_none", "share_only_weights", "share_only_bias"])
def test_shared_weight(temp_files, outer_flat, inner_flat, sharing):
    """Test FSDP with a model with shared weights."""

    outer_flat = outer_flat == "outer_flat"
    inner_flat = inner_flat == "inner_flat"
    world_size = 2

    # Get reference results.
    model = Model(sharing=sharing)
    sd_before = deepcopy(model.state_dict())
    in_data = torch.rand(1, 4).cuda()
    _train(model, in_data, world_size)
    sd_after = deepcopy(model.state_dict())
    # Before and after state should not be equal.
    assert not objects_are_equal(sd_before, sd_after)

    # Save data
    torch.save(sd_before, temp_files[2])
    torch.save(sd_after, temp_files[3])
    torch.save(in_data, temp_files[4])

    # Run FSDP
    mp.spawn(
        _dist_worker,
        (world_size, temp_files, outer_flat, inner_flat, sharing),
        nprocs=world_size,
    )


def _dist_worker(rank, world_size, files, outer_flat, inner_flat, sharing):

    # Get data from files.
    file1, file2, sd_before, sd_after, in_data = files
    sd_before = torch.load(sd_before, map_location=lambda storage, loc: storage.cuda(rank))
    sd_after = torch.load(sd_after, map_location=lambda storage, loc: storage.cuda(rank))
    in_data = torch.load(in_data, map_location=lambda storage, loc: storage.cuda(rank))

    result = dist_init(rank=rank, world_size=world_size, filename=file1, filename_rpc=file2)
    assert result, "Dist init failed"

    fsdp_model = FSDP(Model(with_fsdp=True, inner_flat=inner_flat, sharing=sharing), flatten_parameters=outer_flat)
    fsdp_model.load_state_dict(sd_before)

    _train(fsdp_model, in_data)

    objects_are_equal(sd_after, fsdp_model.state_dict(), raise_exception=True)

    teardown()


def _train(model, in_data, steps_per_iter=1):
    optim = SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        # Simulate multiple ranks.
        for _ in range(steps_per_iter):
            out = model(in_data)
            out.sum().backward()
        # Simulate gradient means between ranks.
        if steps_per_iter > 1:
            with torch.no_grad():
                for p in model.parameters():
                    p.grad /= steps_per_iter
        optim.step()
        model.zero_grad(set_to_none=True)
