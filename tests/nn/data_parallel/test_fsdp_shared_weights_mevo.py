# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with shared weights between wrappers using a model with mevo kernel. """

from copy import deepcopy

import pytest
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.optim import SGD

from fairscale.experimental.nn import MEVO
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.utils.testing import dist_init, objects_are_equal, skip_if_single_gpu, teardown, temp_files_ctx

VOCAB = 4
D_MODEL = 2
BS = 2
SEQ = 3
TILE = 2

_large = True

if _large:
    VOCAB = 1024 * 50
    D_MODEL = 1024
    BS = 2
    SEQ = 16
    TILE = 16


class Model(nn.Module):
    def __init__(self, with_fsdp=False, wrap_middle="none"):
        super().__init__()
        self.l0 = nn.Embedding(VOCAB, D_MODEL).cuda().half()
        nn.init.uniform_(self.l0.weight, -1.0e-1, 1.0e-1)
        self.l1 = MEVO(self.l0.weight, tile_factor=TILE, reduction="sum")
        self.middle = nn.Linear(D_MODEL, D_MODEL).cuda().half()
        # LNs are not strictly needed for this test, but they help reduce the loss quickly
        # and improves the numerical stability.
        self.ln1 = nn.LayerNorm(D_MODEL).cuda().half()
        self.ln2 = nn.LayerNorm(D_MODEL).cuda().half()

        if with_fsdp:
            # Shared layers much be un-flatten.
            self.l0 = FSDP(self.l0, flatten_parameters=False, mixed_precision=False, compute_dtype=torch.float16)
            self.l1 = FSDP(self.l1, flatten_parameters=False, mixed_precision=False, compute_dtype=torch.float16)
            self.l1.append_shared_param(self.l0.module.weight)
            # These are for debugging.
            # print(id(self.l0), "is emb")
            # print(id(self.l1), "is out")
            assert wrap_middle in ["none", "flat", "nonflat"]
            if wrap_middle != "none":
                self.middle = FSDP(
                    self.middle,
                    flatten_parameters=wrap_middle == "flat",
                    mixed_precision=False,
                    compute_dtype=torch.float16,
                )
                # print(id(self.middle), "is middle")

    def forward(self, x):
        target = x + 1
        x = self.l0(x)
        x = self.ln1(x)
        x = self.middle(x)
        x = self.ln2(x)
        x = self.l1(x, target)
        print("LOSS", x.item())
        assert x.item() not in [float("-inf"), float("inf")]
        return x


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    # dist_init needs 2 files + 3 files for before state, after state, in_data.
    with temp_files_ctx(5) as files:
        yield files


@skip_if_single_gpu
@pytest.mark.parametrize("wrap_middle", ["none", "flat", "nonflat"])
def test_shared_weight_mevo(temp_files, wrap_middle):
    """Test FSDP with a model with shared weights."""

    # from fairscale.utils import torch_version
    # if torch_version() < (1, 9, 0):
    #    pytest.skip("only support 1.9+")

    world_size = 2

    # Get ref.
    model = Model()
    sd_before = deepcopy(model.state_dict())
    in_data = (torch.rand(BS, SEQ) * (VOCAB - 1)).cuda().long()
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
        _dist_worker, (world_size, temp_files, wrap_middle), nprocs=world_size,
    )


def _dist_worker(rank, world_size, files, wrap_middle):

    # Get data from files.
    file1, file2, sd_before, sd_after, in_data = files
    sd_before = torch.load(sd_before, map_location=lambda storage, loc: storage.cuda(rank))
    sd_after = torch.load(sd_after, map_location=lambda storage, loc: storage.cuda(rank))
    in_data = torch.load(in_data, map_location=lambda storage, loc: storage.cuda(rank))

    result = dist_init(rank=rank, world_size=world_size, filename=file1, filename_rpc=file2)
    assert result, "Dist init failed"

    fsdp_model = FSDP(
        # To debug: first make with_fsdp=False (no inner wrapping) work, then enable inner wrapping
        # and make that work.
        Model(with_fsdp=True, wrap_middle=wrap_middle),
        flatten_parameters=False,
        mixed_precision=False,
        compute_dtype=torch.float16,
    )
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
            out.backward()
        # Simulate gradient means between ranks.
        if steps_per_iter > 1:
            with torch.no_grad():
                for p in model.parameters():
                    p.grad /= steps_per_iter
        with torch.no_grad():
            for p in model.parameters():
                assert not torch.isinf(p.grad).any() and not torch.isnan(p.grad).any()
        optim.step()
        model.zero_grad(set_to_none=True)
