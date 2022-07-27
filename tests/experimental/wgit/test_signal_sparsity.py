# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fair_dev.testing.testing import objects_are_equal
from fairscale.experimental.wgit.signal_sparsity import SignalSparsity


def get_test_params():
    """Helper function to create and return a list of tuples of the form:
    (in_tensor, expected_tensor, dim, percent, top_k_element) to be used as parameters for tests.
    """
    # input in_tensors
    tensor_4x3 = torch.arange(12).reshape(4, 3)
    tensor_2x2x3 = torch.arange(12).reshape(3, 2, 2)

    # Expected SST output tensors for 4x3 tensor of ascending ints
    expected_4x3_None = torch.tensor(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],  # with dim=None, top-2
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [21.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [30.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )

    expected_4x3_0 = torch.tensor(
        [
            [0.0000000 + 0.0000000j, 0.0000000 + 0.0000000j, 0.0000000 + 0.0000000j],  # with dim=0, top-2
            [0.0000000 + 0.0000000j, 0.0000000 + 0.0000000j, 0.0000000 + 0.0000000j],
            [21.0000000 + 0.0000000j, -1.5000000 + 0.8660254j, -1.5000000 - 0.8660254j],
            [30.0000000 + 0.0000000j, -1.5000000 + 0.8660254j, -1.5000000 - 0.8660254j],
        ],
        dtype=torch.complex64,
    )

    expected_4x3_1 = torch.tensor(
        [
            [3.0000000 + 0.0000000j, -1.5000000 + 0.8660254j, 0.0000000 + 0.0000000j],  # with dim=1, top-2
            [12.0000000 + 0.0000000j, -1.5000000 + 0.8660254j, 0.0000000 + 0.0000000j],
            [21.0000000 + 0.0000000j, -1.5000000 + 0.8660254j, 0.0000000 + 0.0000000j],
            [30.0000000 + 0.0000000j, -1.5000000 + 0.8660254j, 0.0000000 + 0.0000000j],
        ],
        dtype=torch.complex64,
    )

    expected_2x2x3_1 = torch.tensor(
        [
            [[1.0 + 0.0j, -1.0 + 0.0j], [5.0 + 0.0j, -1.0 + 0.0j]],  # with dim=1, top-2
            [[9.0 + 0.0j, -1.0 + 0.0j], [13.0 + 0.0j, -1.0 + 0.0j]],
            [[17.0 + 0.0j, -1.0 + 0.0j], [21.0 + 0.0j, -1.0 + 0.0j]],
        ],
        dtype=torch.complex64,
    )

    return [
        (tensor_4x3, expected_4x3_None, None, 20, 2),
        (tensor_4x3, expected_4x3_0, 0, 50, 2),
        (tensor_4x3, expected_4x3_1, 1, 70, 2),
        (tensor_2x2x3, expected_2x2x3_1, 1, 100, 2),
    ]


def get_valid_conf_arg_list():
    """Returns a map object of keyword arguments (as dicts) to be used as parameters for test_validate_conf."""

    def kwargs(vals_list):
        """Maps the values in input vals_list to the keys in arg_key_list"""

        arg_key_list = [
            "sst_top_k_element",
            "sst_top_k_percent",
            "sst_top_k_dim",
            "dst_top_k_element",
            "dst_top_k_percent",
            "dst_top_k_dim",
        ]
        return dict(zip(arg_key_list, vals_list))

    # Validate value error is raised when, either:
    # 1. One and only one of sst (or dst) percent and element is not provided a value (not None).
    # 2. Both of sst (or dst) percent and element is set to None.
    # 3. top_k_percent and top_k_element are not in valid range (elem > 0) and for 0 < percent <= 100.
    element = 10
    percent = 50
    dim = 0
    args_list = [
        [element, percent, dim, element, None, dim],  # case 1.
        [element, None, dim, element, percent, dim],
        [None, None, dim, element, None, dim],  # case 2.
        [element, None, dim, None, None, dim],
        [0, None, dim, None, None, dim],  # case 3.
        [None, 0, dim, None, None, dim],
        [element, None, dim, 0, None, dim],
        [element, None, dim, None, 0, dim],
    ]
    return map(kwargs, args_list)


@pytest.mark.parametrize("kwargs", get_valid_conf_arg_list())
def test_validate_conf(kwargs):
    """Validate value error is raised with each kwargs returned by get_valid_conf_arg_list"""
    pytest.raises(ValueError, SignalSparsity, **kwargs)


@pytest.mark.parametrize(
    "tensor, dim",
    [
        (torch.arange(20).reshape(4, 5), None),
        (torch.arange(20).reshape(4, 5), 0),
        (torch.arange(20).reshape(4, 5), 1),
        (torch.arange(80).reshape(4, 5, 4), None),
        (torch.arange(80).reshape(4, 5, 4), 0),
        (torch.arange(80).reshape(4, 5, 4), 1),
        (torch.arange(80).reshape(4, 5, 4), 2),
    ],
)
def test_dense_to_sst_perfect_recons(tensor, dim):
    """Tests the dense_to_sst method whether it simply performs an FFT transformation
    when top_k_percent is set at 100.
    """
    sparser_2d = SignalSparsity(sst_top_k_percent=100, sst_top_k_dim=dim, dst_top_k_percent=100)
    assert all((sparser_2d.dense_to_sst(tensor) == torch.fft.fft(tensor)).flatten())


@pytest.mark.parametrize("tensor, expected, dim, percent, k", get_test_params())
def test_dense_to_sst_fixed(tensor, expected, dim, percent, k):
    """Tests for fixed input dense tensor and fixed expected output SST tensor for top-2 elements."""
    sparser_2d = SignalSparsity(sst_top_k_percent=None, sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_percent=100)
    sst = sparser_2d.dense_to_sst(tensor)
    objects_are_equal(sst, expected, raise_exception=True)


@pytest.mark.parametrize("tensor, expected, dim, percent, k", get_test_params())
def test_percent_element(tensor, expected, dim, percent, k):
    """Tests whether comparative values for top_k_element and top_k_percent returns same outputs"""
    sparser_2d = SignalSparsity(sst_top_k_percent=None, sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_percent=100)
    sst_element = sparser_2d.dense_to_sst(tensor)

    sparser_2d = SignalSparsity(
        sst_top_k_percent=percent, sst_top_k_element=None, sst_top_k_dim=dim, dst_top_k_percent=100
    )
    sst_percent = sparser_2d.dense_to_sst(tensor)
    objects_are_equal(sst_element, sst_percent, raise_exception=True)
