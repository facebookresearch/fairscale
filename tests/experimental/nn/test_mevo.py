# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import os

import pytest
import torch

from fairscale.experimental.nn import MEVO
from fairscale.experimental.nn.mevo import BaselineSoftmaxNllLoss, get_data
from fairscale.utils import torch_version
from fairscale.utils.testing import skip_if_no_cuda


@pytest.fixture(scope="session", params=[torch.float16, torch.float32])
def input_data(request):
    shape = ((2, 3), (3, 4))
    return get_data(shape, dtype=request.param)


_dense_out = {}  # type: ignore
_dense_grad = {}  # type: ignore


@skip_if_no_cuda
def test_torch_fuse_all():
    """Test the MEVO kernel by itself."""

    if torch_version() < (1, 9, 0):
        pytest.skip("only support 1.9+")

    torch.random.manual_seed(os.getpid())
    shape = ((5, 3), (3, 7))
    large = True
    if large:
        shape = ((1 * 2048, 4096), (4096, 256008))
    print("\nshapes are", shape)

    input, weight, target = get_data(shape, dtype=torch.float16)
    k = MEVO(weight, tile_factor=16)

    o = k(input, target)
    o.backward()
    print(o, o.shape)
    del o

    cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
    mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
    print("cur and peak mem for tiled fwd+bwd =", cur_mem, mem)

    assert input.shape == input.grad.shape
    input_data = input.data.cpu()
    input_grad1 = input.grad.cpu()
    del input

    cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
    mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
    print("after moving input and its grad, cur and peak mem for tiled fwd+bwd =", cur_mem, mem)

    print(weight.grad.norm(), weight.grad)
    g1 = weight.grad.clone()
    weight.grad = None

    input = input_data.cuda().requires_grad_(True)
    refk = BaselineSoftmaxNllLoss(weight)
    o = refk(input, target)
    o.backward()
    print(o, o.shape)
    del o
    print(weight.grad.norm(), weight.grad)
    g2 = weight.grad.clone()
    input_grad2 = input.grad.cpu()

    diff = g1 - g2
    print("weight grad diff", diff.min(), diff.max())
    diff = input_grad1 - input_grad2
    print("input grad diff", diff.min(), diff.max())
