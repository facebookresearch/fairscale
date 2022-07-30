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
    (dense, expected_sst, expected_dst, expected_reconstructed_tensor (RT), dim, percent, top_k_element)
    to be used as parameters for tests.
    """
    # input in_tensors
    tensor_4x3_None = torch.arange(12).reshape(4, 3).float()
    tensor_4x3_0 = torch.arange(50, 62).reshape(4, 3) / 100
    tensor_3x3_1 = torch.linspace(-5, 5, 9).reshape(3, 3)
    tensor_2x2x3 = torch.arange(12).reshape(3, 2, 2).float()

    # with dim=None, top-2
    expd_sst_4x3_None = torch.tensor(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [21.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [30.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )

    # with dim=None, top-2
    expd_dst_4x3_None = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 4.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
    )

    # expected_reconstructed_tensor with dim=None and top-2 for both sst and dst
    expd_rt_4x3_None = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 4.0, 5.0], [7.0, 7.0, 7.0], [10.0, 10.0, 10.0]], dtype=torch.float32
    )

    # with dim=0, top-2
    expd_sst_4x3_0 = torch.tensor(
        [
            [0.0000000000 + 0.0000000000j, 0.0000000000 + 0.0000000000j, 0.0000000000 + 0.0000000000j],
            [0.0000000000 + 0.0000000000j, -0.0150000453 + 0.0086602457j, -0.0150000453 - 0.0086602457j],
            [1.7100000381 + 0.0000000000j, 0.0000000000 + 0.0000000000j, 0.0000000000 + 0.0000000000j],
            [1.7999999523 + 0.0000000000j, -0.0150000453 + 0.0086602457j, -0.0150000453 - 0.0086602457j],
        ],
        dtype=torch.complex64,
    )

    # with dim=0, top-2
    expd_dst_4x3_0 = torch.tensor(
        [[0.5000, 0.5100, 0.5200], [0.5400, 0.5400, 0.5400], [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]]
    )

    # expected_reconstructed_tensor with dim=0 and top-2 for both sst and dst
    expd_rt_4x3_0 = torch.tensor(
        [[0.5000, 0.5100, 0.5200], [0.5300, 0.5400, 0.5500], [0.5700, 0.5700, 0.5700], [0.5900, 0.6000, 0.6100]]
    )

    # with dim=1, top-2
    expd_sst_3x3_1 = torch.tensor(
        [
            [-11.2500000000 + 0.0000000000j, -1.8750000000 + 1.0825316906j, 0.0000000000 + 0.0000000000j],
            [0.0000000000 + 0.0000000000j, -1.8750000000 + 1.0825316906j, -1.8750000000 - 1.0825316906j],
            [11.2500000000 + 0.0000000000j, -1.8750000000 + 1.0825316906j, 0.0000000000 + 0.0000000000j],
        ],
        dtype=torch.complex64,
    )

    # with dim=1, top-2
    expd_dst_3x3_1 = torch.tensor(
        [
            [-6.2500000000e-01, 0.0000000000e00, 6.2500000000e-01],
            [0.0000000000e00, -4.8244856998e-08, 0.0000000000e00],
            [-6.2500000000e-01, 0.0000000000e00, 6.2500000000e-01],
        ]
    )

    # expected_reconstructed_tensor with dim=1 and top-2 for both sst and dst
    expd_rt_3x3_1 = torch.tensor([[-5.0000, -3.7500, -2.5000], [-1.2500, 0.0000, 1.2500], [2.5000, 3.7500, 5.0000]])

    # with dim=1, top-1
    expd_sst_2x2x3_1 = torch.tensor(
        [
            [[0.0 + 0.0j, -1.0 + 0.0j], [5.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, -1.0 + 0.0j], [13.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, -1.0 + 0.0j], [21.0 + 0.0j, 0.0 + 0.0j]],
        ],
        dtype=torch.complex64,
    )

    # with dim=1, top-1
    expd_dst_2x2x3_1 = torch.tensor(
        [
            [[0.5000, 0.5000], [0.0000, 0.0000]],
            [[4.5000, 4.5000], [0.0000, 0.0000]],
            [[8.5000, 8.5000], [0.0000, 0.0000]],
        ],
        dtype=torch.float32,
    )

    # expected_reconstructed_tensor with dim=1 and top-1 for both sst and dst
    expd_rt_2x2x3_1 = torch.tensor(
        [
            [[0.0000, 1.0000], [2.5000, 2.5000]],
            [[4.0000, 5.0000], [6.5000, 6.5000]],
            [[8.0000, 9.0000], [10.5000, 10.5000]],
        ],
        dtype=torch.float32,
    )

    return [
        (tensor_4x3_None, expd_sst_4x3_None, expd_dst_4x3_None, expd_rt_4x3_None, None, 20, 2),
        (tensor_4x3_0, expd_sst_4x3_0, expd_dst_4x3_0, expd_rt_4x3_0, 0, 50, 2),
        (tensor_3x3_1, expd_sst_3x3_1, expd_dst_3x3_1, expd_rt_3x3_1, 1, 70, 2),
        (tensor_2x2x3, expd_sst_2x2x3_1, expd_dst_2x2x3_1, expd_rt_2x2x3_1, 1, 50, 1),
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


@pytest.mark.parametrize("tensor, expd_sst, unused1, unused2, dim, unused3, k", get_test_params())
def test_dense_to_sst_fixed(tensor, expd_sst, unused1, unused2, dim, unused3, k):
    """Tests for fixed input dense tensor and fixed expected output SST tensor."""
    sparser_2d = SignalSparsity(sst_top_k_percent=None, sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_percent=100)
    sst = sparser_2d.dense_to_sst(tensor)
    objects_are_equal(sst, expd_sst, raise_exception=True)


@pytest.mark.parametrize("tensor, unused1, unused2, unused3, dim, percent, k", get_test_params())
def test_percent_element(tensor, unused1, unused2, unused3, dim, percent, k):
    """Tests whether comparative values for top_k_element and top_k_percent returns same outputs"""
    sparser_2d = SignalSparsity(sst_top_k_percent=None, sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_percent=100)
    sst_element = sparser_2d.dense_to_sst(tensor)

    sparser_2d = SignalSparsity(
        sst_top_k_percent=percent, sst_top_k_element=None, sst_top_k_dim=dim, dst_top_k_percent=100
    )
    sst_percent = sparser_2d.dense_to_sst(tensor)
    objects_are_equal(sst_element, sst_percent, raise_exception=True)


@pytest.mark.parametrize("tensor, sst, expd_dst, unused1, dim, unused2, k", get_test_params())
def test_dense_sst_to_dst(tensor, sst, expd_dst, unused1, dim, unused2, k):
    """Tests fixed expected output DST tensor with fixed input dense and SST tensors."""
    sparser_2d = SignalSparsity(sst_top_k_percent=None, sst_top_k_element=k, dst_top_k_element=k, dst_top_k_dim=dim)
    dst = sparser_2d.dense_sst_to_dst(tensor, sst)
    objects_are_equal(dst, expd_dst, raise_exception=True)


@pytest.mark.parametrize(
    "dense, k, dim",
    [
        (torch.linspace(0.01, 0.06, 40).reshape(5, 8), 40, None),  # top-40, dim=None
        (torch.linspace(0.1, 0.6, 30).reshape(5, 6), 5, 0),  # top-5, dim=0
        (torch.linspace(-0.1, 0.6, 35).reshape(7, 5), 5, 1),  # top-5, dim=1
        (torch.arange(60).float().reshape(10, 6), 60, None),  # top-60, dim=None
        (torch.arange(60).float().reshape(10, 6), 10, 0),  # top-10, dim=0
        (torch.arange(60).float().reshape(10, 6), 6, 1),  # top-6, dim=1
        (torch.arange(60).float().reshape(2, 5, 6), 5, 1),  # top-5, dim=1
    ],
)
def test_sst_dst_to_perfect_dense_reconstruction(dense, k, dim):
    """Tests whether perfect reconstruction of input dense tensor is generated when top-k matches the numel
    across some dimension dim for both SST and DST.
    """
    sparser = SignalSparsity(sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_element=k, dst_top_k_dim=dim)
    sst = sparser.dense_to_sst(dense)
    dst = sparser.dense_sst_to_dst(dense, sst)
    dense_recons = sparser.sst_dst_to_dense(sst, dst)
    objects_are_equal(dense, dense_recons, raise_exception=True)


@pytest.mark.parametrize("unused1, sst, dst, expd_rt, dim, unused2, k", get_test_params())
def test_sst_dst_to_dense(unused1, sst, dst, expd_rt, dim, unused2, k):
    """Tests the correct expected reconstruction from frozen sst and dst tensors."""
    sparser = SignalSparsity(sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_element=k, dst_top_k_dim=dim)
    dense_recons = sparser.sst_dst_to_dense(sst, dst)
    objects_are_equal(dense_recons, expd_rt, raise_exception=True)
