# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairscale.experimental.wgit.signal_sparsity import SignalSparsity, _get_k_for_topk, _top_k_total_size

TRIALS = 5
DIM_LIST = [None, 0, 1, 2]
SPARSITY_ATOL = 8e-2
FFT_RECONS_ATOL = 2e-3


@pytest.fixture
def sst_and_default_sps():
    """A fixture returning a function that calculates the sst from pytorch operations
    and compares it with the sst from dense_to_sst method."""

    def assert_and_get_sps_sst(tensor, percent, element, dim):
        dense_freq = torch.fft.fft(tensor)

        real_dense_freq = dense_freq.real
        sparser_2d = SignalSparsity(
            sst_top_k_percent=percent, sst_top_k_element=element, sst_top_k_dim=dim, dst_top_k_percent=100
        )
        top_k_total_size = _top_k_total_size(tensor, dim)
        k = _get_k_for_topk(percent, element, top_k_total_size)
        default_sst = torch.zeros_like(dense_freq)

        orig_shape = dense_freq.shape
        if dim is None and len(orig_shape) > 1:
            default_sst = default_sst.reshape(-1)
            real_dense_freq = real_dense_freq.reshape(-1)
            dense_freq = dense_freq.reshape(-1)
            dim = -1

        _, i = real_dense_freq.abs().topk(k, dim=dim)
        default_sst = default_sst.scatter(dim, i, dense_freq.gather(dim, i)).reshape(orig_shape)

        # verify if sst only has top-k values across the right dim
        dim = None if (dim == -1) else dim
        assert all((default_sst.abs().count_nonzero(dim) == k).flatten())

        sst = sparser_2d.dense_to_sst(tensor)
        assert all((default_sst == sst).flatten())
        assert sst.shape == tensor.shape
        return default_sst, sst

    return assert_and_get_sps_sst


def test_validate_conf():
    """Tests the config validation for the Signal Sparsity class."""

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

    # Validate assertion error is raised when, either
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
    for args in args_list:
        pytest.raises(ValueError, SignalSparsity, **kwargs(args))


def test_dense_to_sst(sst_and_default_sps):
    """Tests the dense_to_sst method with fixed inputs"""
    tensors = list()
    tensors.append(torch.arange(20).reshape(4, 5))
    tensors.append(torch.arange(80).reshape(4, 5, 4))

    def elem_percent_equality(tensor, percent, element, dim):
        """verifies if the topk element and top k percent returns the same results
        for same sparsity in a fixed tensor along some dim.
        """
        _, sst_percent = sst_and_default_sps(tensor, percent, None, dim=dim)
        _, sst_elem = sst_and_default_sps(tensor, None, element, dim=dim)
        assert all((sst_percent == sst_elem).flatten())

    for tensor in tensors:
        dim_list = [None] + list(range(tensor.dim()))
        for dim in dim_list:
            # Should return the FFT transformed tensor with top_100_percent for sst
            sparser_2d = SignalSparsity(sst_top_k_percent=100, sst_top_k_dim=dim, dst_top_k_percent=100)
            assert all((sparser_2d.dense_to_sst(tensor) == torch.fft.fft(tensor)).flatten())

            # Test for different sparsity levels using top_k_percent and top_k_elements
            for percent, element in zip([20, 40, 60, 80], [1, 2, 3, 4]):
                sst_and_default_sps(tensor, percent=percent, element=None, dim=dim)
                sst_and_default_sps(tensor, percent=None, element=element, dim=dim)

            # Test that both top_k_percent and top_k_element returns same results
            if dim is None:
                percent_l = [25, 50, 75]
                elem_l = [
                    tensor.numel() * percent_l[0] / 100,
                    tensor.numel() * percent_l[1] / 100,
                    tensor.numel() * percent_l[2] / 100,
                ]
                elem_l = map(int, elem_l)
                for percent, elem in zip(percent_l, elem_l):
                    elem_percent_equality(tensor, percent, elem, dim)
            else:
                for percent, elem in zip([25, 50, 75], [1, 2, 3]):
                    elem_percent_equality(tensor, percent, elem, dim)
