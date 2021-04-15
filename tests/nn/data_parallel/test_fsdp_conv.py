# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" This is a test to validate pytorch behavior with respect to what FSDP
    is potentially doing for mixed precision computing with conv layers.

    Different pytorch/cuda version seems to behave differently.
"""

import random

import torch

from fairscale.utils.testing import skip_if_no_cuda


@skip_if_no_cuda
def test1():
    if random.randint(0, 1) == 0:
        print("setting benchmark = True")
        torch.backends.cudnn.benchmark = True
    if random.randint(0, 1) == 0:
        print("setting determinism = True")
        if hasattr(torch, "_set_deterministic"):
            torch._set_deterministic(True)
        else:
            torch.use_deterministic_algorithms(True)

    # Get a big tensor on cuda in fp16.
    big_tensor = torch.rand(1952152).cuda().half()

    # Get their weights.
    i = 0
    a_w = big_tensor[i : 528 * 528]
    a_w = a_w.reshape((528, 528, 1, 1))
    i += a_w.numel()

    b_w = big_tensor[i : i + 528 * 264 * 3 * 3]
    b_w = b_w.reshape((528, 264, 3, 3))
    i += b_w.numel()

    se0_w = big_tensor[i : i + 132 * 528]
    se0_w = se0_w.reshape((132, 528, 1, 1))
    i += se0_w.numel()
    se0_b = big_tensor[i : i + 132]
    i += se0_b.numel()

    se2_w = big_tensor[i : i + 528 * 132]
    se2_w = se2_w.reshape((528, 132, 1, 1))
    i += se0_w.numel()
    se2_b = big_tensor[i : i + 528]
    i += se0_b.numel()

    c_w = big_tensor[i : i + 528 * 528]
    c_w = c_w.reshape((528, 528, 1, 1))
    i += c_w.numel()

    # Get the convs accordingly.
    a = torch.nn.Conv2d(528, 528, (1, 1), bias=False).cuda()
    a.weight.data = a_w

    b = torch.nn.Conv2d(528, 264, (3, 3), bias=False, groups=2).cuda()
    b.weight.data = b_w

    se0 = torch.nn.Conv2d(132, 528, (1, 1), bias=True).cuda()
    se0.weight.data = se0_w
    se0.bias.data = se0_b

    se2 = torch.nn.Conv2d(528, 132, (1, 1), bias=True).cuda()
    se2.weight.data = se2_w
    se2.bias.data = se2_b

    c = torch.nn.Conv2d(528, 528, (1, 1), bias=False).cuda()
    c.weight.data = c_w

    # Run forward.
    i = torch.rand(4, 528, 56, 56).cuda().half()
    for layer in [a, b, se0, se2, c]:
        i = layer(i)
