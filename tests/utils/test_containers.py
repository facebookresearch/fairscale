# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test utility classes from containers.py. """

import random

import pytest
import torch

from fairscale.utils.containers import (
    apply_to_tensors,
    pack_kwargs,
    split_non_tensors,
    unpack_kwargs,
    unpack_non_tensors,
)


@pytest.mark.parametrize("devices", [["cpu"], ["cuda"], ["cpu", "cuda"]])
def test_apply_to_tensors(devices):
    """Test apply_to_tensors for both cpu & gpu"""
    if "cuda" in devices and not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("Skipped due to lack of GPU")
    expected = 0

    def get_a_tensor():
        """Return a random tensor on random device."""
        dev = random.choice(devices)
        shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
        t = torch.rand(shape).to(dev)
        nonlocal expected
        expected += t.numel()
        return t

    # create a mixed bag of data.
    data = [1, "str"]
    data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
    data.insert(0, set(["x", get_a_tensor(), get_a_tensor()]))
    data.append(([1], get_a_tensor(), (1), [get_a_tensor()], set((1, 2))))

    total = 0

    def fn(t, x=[[total]]):
        nonlocal total
        total += t.numel()
        return t

    apply_to_tensors(fn, data)
    assert total == expected, f"{total} vs. {expected}"


def test_pack_unpack():
    # tbd
    p = pack_kwargs
    up = unpack_kwargs


def test_split_unpack():
    # tbd
    s = split_non_tensors
    up = unpack_non_tensors
