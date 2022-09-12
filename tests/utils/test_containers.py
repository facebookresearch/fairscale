# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test utility classes from containers.py. """

from collections import OrderedDict, namedtuple
import random

import pytest
import torch
import torch.nn as nn

from fairscale.internal.containers import (
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
        shape = random.choice((1, (2, 3), (4, 5, 6), (7, 8, 9, 10)))
        t = torch.rand(shape).to(dev)
        nonlocal expected
        expected += t.numel()
        return t

    # create a mixed bag of data.
    data = [1, "str"]  # list
    # dict
    data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
    # set
    data.insert(0, set(["x", get_a_tensor(), get_a_tensor()]))
    # tuple
    data.append(([1], get_a_tensor(), 1, [get_a_tensor()], set((1, 2))))
    # OrderedDict
    od = OrderedDict()
    od["k"] = "value"
    data.append(od)
    # namedtuple
    NT = namedtuple("NT", ["key1", "key2"])
    nt = NT(key1=1, key2=get_a_tensor())
    data.append(nt)

    total = 0

    def fn(t, x=[[total]]):
        nonlocal total
        total += t.numel()
        return t

    new_data = apply_to_tensors(fn, data)
    assert total == expected, f"{total} vs. {expected}"
    for i, v in enumerate(data):
        assert type(new_data[i]) == type(v), f"expected type {type(v)} got {type(new_data[i])}"


def test_pack_unpack():
    """Test pack_kwargs and unpack_kwargs."""
    kwarg_keys, flat_args = pack_kwargs(1, 2, 3, 4)
    assert kwarg_keys == tuple()
    assert flat_args == (1, 2, 3, 4)

    kwarg_keys, flat_args = pack_kwargs(a=1, b={2: "2"}, c={3}, d=[4], e=(5,))
    assert kwarg_keys == ("a", "b", "c", "d", "e")
    assert flat_args == (1, {2: "2"}, {3}, [4], (5,))

    kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
    assert kwarg_keys == ("a", "b")
    assert flat_args == (1, 2, 3, 4)

    args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
    assert args == (1, 2)
    assert kwargs == {"a": 3, "b": 4}

    args, kwargs = unpack_kwargs([], flat_args)
    assert kwargs == {}
    assert args == (1, 2, 3, 4)

    args, kwargs = unpack_kwargs(["a", "b", "c", "d"], flat_args)
    assert kwargs == {"a": 1, "b": 2, "c": 3, "d": 4}
    assert args == tuple()

    with pytest.raises(AssertionError):
        # too many keys should assert.
        args, kwargs = unpack_kwargs(["a", "b", "c", "d", "e"], flat_args)


def test_split_unpack():
    """Test split_non_tensors and unpack_non_tensors."""
    x = torch.Tensor([1])
    y = torch.Tensor([2])

    # degenerate case, args is a single tensor.
    tensors, packed_non_tensors = split_non_tensors(x)
    assert tensors == (x,)
    assert packed_non_tensors is None

    tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
    assert tensors == (x, y)
    assert packed_non_tensors == {
        "is_tensor": [True, True, False, False],
        "objects": [None, 3],
    }
    recon = unpack_non_tensors(tensors, packed_non_tensors)
    assert recon == (x, y, None, 3)

    tensors, packed_non_tensors = split_non_tensors((None, 3, x, y))
    recon = unpack_non_tensors(tensors, packed_non_tensors)
    assert recon == (None, 3, x, y)

    tensors, packed_non_tensors = split_non_tensors((None, 3))
    recon = unpack_non_tensors(tensors, packed_non_tensors)
    assert recon == (None, 3)

    tensors, packed_non_tensors = split_non_tensors((x, y))
    recon = unpack_non_tensors(tensors, packed_non_tensors)
    assert recon == (x, y)

    recon = unpack_non_tensors(tensors, None)
    assert recon == (x, y)

    with pytest.raises(AssertionError):
        # assert the second arg should be a dict.
        recon = unpack_non_tensors(tensors, set())

    with pytest.raises(AssertionError):
        # assert the content of the second arg should be sane.
        recon = unpack_non_tensors(tensors, {"is_tensor": [], "objects": []})


def test_packed_sequence():
    """Test to ensure RNN packed sequences are modified correctly."""
    rnn = nn.RNN(5, 5)

    x = torch.rand((5, 1, 5), dtype=torch.float)
    seq_length = torch.tensor([4], dtype=torch.int)

    def fill_fn(x):
        x.fill_(0)

    x = nn.utils.rnn.pack_padded_sequence(x, seq_length)
    x, h = rnn(x)
    x = apply_to_tensors(fill_fn, x)
    x, _ = nn.utils.rnn.pad_packed_sequence(x)
    assert torch.sum(x) == 0
