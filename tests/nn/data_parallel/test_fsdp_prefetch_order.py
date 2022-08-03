# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Check that the ordering used for prefetching model weights
    matches the expected execution order for the model.
"""

import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import SGD

from fair_dev.testing.testing import dist_init, skip_if_single_gpu, teardown
from fairscale.internal import torch_version
from fairscale.nn import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState, auto_wrap_bn
from fairscale.nn.wrap import enable_wrap, wrap


def _get_module_type(fsdp):
    m = fsdp._fsdp_wrapped_module._fpw_module
    if isinstance(m, nn.Sequential):
        return type(m[0])
    return type(m)


def _test_func(rank, world_size, fsdp_config, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)

    torch.cuda.set_device(rank)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=True))
            self.block2 = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=False))
            self.block3 = nn.Linear(12, 8)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(4, 10))

        def forward(self, x):
            return self.head(self.block3(self.block2(self.block1(x))))

    model = Model()
    # Wrapping BatchNorm as separate modules for the forward pass.
    model.block1 = auto_wrap_bn(model.block1, fsdp_config={"reshard_after_forward": True})
    model.block2 = auto_wrap_bn(model.block2, fsdp_config={"reshard_after_forward": True})

    # Checkpoints shouldn't affect the ordering.
    model.block2 = checkpoint_wrapper(model.block2)

    with enable_wrap(
        wrapper_cls=FSDP,
    ):
        model.block1 = wrap(model.block1)
        model.block2 = wrap(model.block2)
        model.block3 = wrap(model.block3)
        model = wrap(model)

    optim = SGD(model.parameters(), lr=0.1)
    model = model.cuda()

    # Orderings are stored in the first pass.
    in_data = torch.randn(size=(2, 3, 16, 16)).cuda()
    in_data.requires_grad = True
    out = model(in_data)
    out.sum().backward()
    optim.step()

    expected_forward_ordering = [nn.BatchNorm2d, nn.Conv2d, nn.BatchNorm2d, nn.Conv2d, nn.Linear, Model]
    actual_forward_ordering = [_get_module_type(m) for m in model._forward_ordering]
    assert expected_forward_ordering == actual_forward_ordering

    expected_backward_ordering = [nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm2d, nn.Conv2d]
    actual_backward_ordering = [_get_module_type(m) for m in model._backward_rebuild_ordering]
    assert expected_backward_ordering == actual_backward_ordering

    model.assert_state(TrainingState.IDLE)
    teardown()


@skip_if_single_gpu
def test():
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    fsdp_config = {}

    # Using world_size > 1 to trigger all-gathers.
    world_size = 2
    mp.spawn(
        _test_func,
        args=(world_size, fsdp_config, temp_file_name, unused),
        nprocs=world_size,
        join=True,
    )
