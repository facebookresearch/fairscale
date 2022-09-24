# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairscale.experimental.wgit.signal_sparsity import SignalSparsity, random_sparse_mask
from fairscale.fair_dev.testing.testing import objects_are_equal

# Our own tolerance
ATOL = 1e-6
RTOL = 1e-5

# enable this for debugging.
# torch.set_printoptions(precision=20)


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
    objects_are_equal(dense, dense_recons, raise_exception=True, rtol=RTOL, atol=ATOL)


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
    # 1. both sst (or dst) percent and element is not provided a value (not None).
    # 2. top_k_percent and top_k_element are not in valid range (elem > 0) and for 0 < percent <= 100.
    element = 10
    percent = 50
    dim = 0
    args_list = [
        [element, percent, dim, element, None, dim],  # case 1.
        [element, None, dim, element, percent, dim],
        [0, None, dim, None, None, dim],  # case 2.
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
    if dim is None:
        fft_tensor = torch.fft.fft(tensor.flatten()).reshape(tensor.shape)
    else:
        fft_tensor = torch.fft.fft(tensor, dim=dim)
    assert all((sparser_2d.dense_to_sst(tensor) == fft_tensor).flatten())


#
# Below are fixed input/output testing.
#


def get_test_params():
    """Helper function to create and return a list of tuples of the form:
    (dense, expected_sst, expected_dst, expected_reconstructed_tensor (RT), dim, percent, top_k_element)
    to be used as parameters for tests.
    """
    # Input tensor 0.
    # We use `sin()` below to make sure the top-2 values are not index sort
    # sensitive. With just `arange()`, we get a linear line and the resulting
    # FFT has many identical second-to-the-largest values. That make top-2 potentially
    # non-deterministic and implementation dependent.
    tensor_4x3_None = torch.arange(12).sin().reshape(4, 3).float()
    # Values are: [[ 0.00000000000000000000,  0.84147095680236816406, 0.90929740667343139648],
    #              [ 0.14112000167369842529, -0.75680249929428100586, -0.95892429351806640625],
    #              [-0.27941548824310302734,  0.65698659420013427734, 0.98935824632644653320],
    #              [ 0.41211849451065063477, -0.54402112960815429688, -0.99999022483825683594]]

    # SST: with dim=None, top-2
    expd_sst_4x3_None = torch.tensor(
        [
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, -1.3618 - 5.7650j],
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, -1.3618 + 5.7650j, 0.0000 + 0.0000j],
        ],
        dtype=torch.complex64,
    )

    # DST: with dim=None, top-2
    expd_dst_4x3_None = torch.tensor(
        [
            [0.22696666419506072998, 0.00000000000000000000, 0.00000000000000000000],
            [0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000],
            [0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000],
            [0.18515183031558990479, 0.00000000000000000000, 0.00000000000000000000],
        ]
    )

    # RT: expected_reconstructed_tensor with dim=None and top-2 for both sst and dst
    expd_rt_4x3_None = torch.tensor(
        [
            [0.00000000000000000000, 0.71862268447875976562, 0.94558942317962646484],
            [0.22696666419506072998, -0.71862268447875976562, -0.94558942317962646484],
            [-0.22696666419506072998, 0.71862268447875976562, 0.94558942317962646484],
            [0.41211849451065063477, -0.71862268447875976562, -0.94558942317962646484],
        ]
    )

    # Input tensor 1.
    tensor_4x3_0 = torch.arange(50, 62).sin().reshape(4, 3) / 100
    # Values are: [[-0.00262374849990010262,  0.00670229177922010422, 0.00986627582460641861],
    #              [ 0.00395925156772136688, -0.00558789074420928955, -0.00999755132943391800],
    #              [-0.00521551026031374931,  0.00436164764687418938, 0.00992872659116983414],
    #              [ 0.00636737979948520660, -0.00304810609668493271, -0.00966117810457944870]]

    # SST: with dim=0, top-1, (use top-1 because top-2 and top-3 would include some identical values)
    expd_sst_4x3_0 = torch.tensor(
        [
            [0.0000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.0000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [-1.81658901274204254150e-02 + 0.0j, 1.96999348700046539307e-02 + 0.0j, 3.94537299871444702148e-02 + 0.0j],
            [0.0000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
        ],
        dtype=torch.complex64,
    )

    # DST: with dim=0, top-2
    expd_dst_4x3_0 = torch.tensor(
        [
            [0.00191772403195500374, 0.00000000000000000000, 0.00000000000000000000],
            [0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000],
            [0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000],
            [0.00000000000000000000, 0.00187687762081623077, 0.00020225439220666885],
        ]
    )

    # RT: expected_reconstructed_tensor with dim=0 and top-2 for both sst and dst
    expd_rt_4x3_0 = torch.tensor(
        [
            [-0.00262374849990010262, 0.00492498371750116348, 0.00986343249678611755],
            [0.00454147253185510635, -0.00492498371750116348, -0.00986343249678611755],
            [-0.00454147253185510635, 0.00492498371750116348, 0.00986343249678611755],
            [0.00454147253185510635, -0.00304810609668493271, -0.00966117810457944870],
        ]
    )

    # Input tensor 2.
    tensor_3x5_1 = torch.Tensor([0, 2, 3, 1, 6, 5, 7, 4, 8, 11, 9, 10, 0, 2, 5]).reshape(3, 5)

    # SST: with dim=1, top-3, because FFT always have symmetric output after the top-1
    expd_sst_3x5_1 = torch.tensor(
        [
            [
                12.00000000000000000000 + 0.00000000000000000000j,
                0,
                -5.23606777191162109375 + 4.25325393676757812500j,
                -5.23606777191162109375 - 4.25325393676757812500j,
                0,
            ],
            [
                35.00000000000000000000 + 0.00000000000000000000j,
                0,
                -5.85410213470458984375 - 1.45308518409729003906j,
                -5.85410213470458984375 + 1.45308518409729003906j,
                0,
            ],
            [
                26.00000000000000000000 + 0.00000000000000000000j,
                12.01722049713134765625 - 3.57971239089965820312j,
                0,
                0,
                12.01722049713134765625 + 3.57971239089965820312j,
            ],
        ],
        dtype=torch.complex64,
    )

    # DST: with dim=1, top-2
    expd_dst_3x5_1 = torch.tensor(
        [
            [
                0.00000000000000000000,
                -1.09442710876464843750,
                0.00000000000000000000,
                0.86524754762649536133,
                0.90557289123535156250,
            ],
            [
                0.00000000000000000000,
                -2.23606777191162109375,
                -1.72360706329345703125,
                0.00000000000000000000,
                2.44721317291259765625,
            ],
            [
                0.00000000000000000000,
                1.95278644561767578125,
                -2.15278673171997070312,
                1.53049504756927490234,
                0.00000000000000000000,
            ],
        ],
    )

    # RT: expected_reconstructed_tensor with dim=1 and top-2 for both sst and dst
    expd_rt_3x5_1 = torch.tensor(
        [
            [
                0.30557289719581604004,
                2.00000000000000000000,
                3.37082076072692871094,
                1.00000000000000000000,
                6.00000000000000000000,
            ],
            [
                4.65835905075073242188,
                7.00000000000000000000,
                4.00000000000000000000,
                6.82917928695678710938,
                11.00000000000000000000,
            ],
            [
                10.00688838958740234375,
                10.00000000000000000000,
                0.00000000000000000000,
                2.00000000000000000000,
                5.32360696792602539062,
            ],
        ]
    )

    # Input tensor 3.
    tensor_3x2x2 = torch.arange(12).cos().reshape(3, 2, 2).float()
    # Values are: [[[ 1.00000000000000000000,  0.54030233621597290039],
    #               [-0.41614684462547302246, -0.98999249935150146484]],
    #             [[-0.65364360809326171875,  0.28366219997406005859],
    #              [ 0.96017026901245117188,  0.75390225648880004883]],
    #             [[-0.14550003409385681152, -0.91113024950027465820],
    #              [-0.83907151222229003906,  0.00442569795995950699]]]

    # SST: with dim=1, top-1
    expd_sst_3x2x2_1 = torch.tensor(
        [
            [[0, 0], [1.41614687442779541016 + 0.0j, 1.53029489517211914062 + 0.0j]],
            [[0, 1.03756451606750488281 + 0.0j], [-1.61381387710571289062 + 0.0j, 0]],
            [[-0.98457157611846923828 + 0.0j, 0], [0, -0.91555595397949218750 + 0.0j]],
        ],
        dtype=torch.complex64,
    )

    # DST: with dim=1, top-1
    expd_dst_3x2x2_1 = torch.tensor(
        [
            [[0.00000000000000000000, -0.22484511137008666992], [0.29192659258842468262, 0.00000000000000000000]],
            [[0.15326333045959472656, -0.23512005805969238281], [0.00000000000000000000, 0.00000000000000000000]],
            [[0.34678575396537780762, -0.45335227251052856445], [0.00000000000000000000, 0.00000000000000000000]],
        ]
    )

    # RT: expected_reconstructed_tensor with dim=1 and top-1 for both sst and dst
    expd_rt_3x2x2_1 = torch.tensor(
        [
            [[0.70807343721389770508, 0.54030233621597290039], [-0.41614684462547302246, -0.76514744758605957031]],
            [[-0.65364360809326171875, 0.28366219997406005859], [0.80690693855285644531, 0.51878225803375244141]],
            [[-0.14550003409385681152, -0.91113024950027465820], [-0.49228578805923461914, 0.45777797698974609375]],
        ]
    )

    return [
        # input, expected sst, dst, rt, sst_dim, percent, top_k.
        (tensor_4x3_None, expd_sst_4x3_None, expd_dst_4x3_None, expd_rt_4x3_None, None, 2 / 12 * 100, 2),
        (tensor_4x3_0, expd_sst_4x3_0, expd_dst_4x3_0, expd_rt_4x3_0, 0, 1 / 3 * 100, 1),
        (tensor_3x5_1, expd_sst_3x5_1, expd_dst_3x5_1, expd_rt_3x5_1, 1, 3 / 5 * 100, 3),
        (tensor_3x2x2, expd_sst_3x2x2_1, expd_dst_3x2x2_1, expd_rt_3x2x2_1, 1, 1 / 2 * 100, 1),
    ]


@pytest.mark.parametrize("tensor, expd_sst, unused1, unused2, dim, unused3, k", get_test_params())
def test_dense_to_sst(tensor, expd_sst, unused1, unused2, dim, unused3, k):
    """Tests for fixed input dense tensor and fixed expected output SST tensor."""
    sparser_2d = SignalSparsity(sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_percent=100)
    sst = sparser_2d.dense_to_sst(tensor)
    objects_are_equal(sst, expd_sst, raise_exception=True, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("tensor, unused1, unused2, unused3, dim, percent, k", get_test_params())
def test_percent_element(tensor, unused1, unused2, unused3, dim, percent, k):
    """Tests whether comparative values for top_k_element and top_k_percent returns same outputs"""
    sparser_2d = SignalSparsity(sst_top_k_percent=None, sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_percent=100)
    sst_element = sparser_2d.dense_to_sst(tensor)

    sparser_2d = SignalSparsity(
        sst_top_k_percent=percent, sst_top_k_element=None, sst_top_k_dim=dim, dst_top_k_percent=100
    )
    sst_percent = sparser_2d.dense_to_sst(tensor)
    objects_are_equal(sst_element, sst_percent, raise_exception=True, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("tensor, sst, expd_dst, unused1, dim, unused2, k", get_test_params())
def test_dense_sst_to_dst(tensor, sst, expd_dst, unused1, dim, unused2, k):
    """Tests fixed expected output DST tensor with fixed input dense and SST tensors."""
    sparser_2d = SignalSparsity(sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_element=k, dst_top_k_dim=dim)
    dst = sparser_2d.dense_sst_to_dst(tensor, sst)
    objects_are_equal(dst, expd_dst, raise_exception=True, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("unused1, sst, dst, expd_rt, dim, unused2, unused3", get_test_params())
def test_sst_dst_to_dense(unused1, sst, dst, expd_rt, dim, unused2, unused3):
    """Tests the correct expected reconstruction from frozen sst and dst tensors."""
    sparser = SignalSparsity(sst_top_k_element=1, sst_top_k_dim=dim, dst_top_k_element=1, dst_top_k_dim=dim)
    dense_recons = sparser.sst_dst_to_dense(sst, dst)
    objects_are_equal(dense_recons, expd_rt, raise_exception=True, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("tensor, expd_sst, expd_dst, expd_rt, dim, unused, k", get_test_params())
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lossy_compress(tensor, expd_sst, expd_dst, expd_rt, dim, unused, k, device):
    """Tests the lossy_compress method against expected sst, dst and reconstruced tensor."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")

    sparser = SignalSparsity(sst_top_k_element=k, sst_top_k_dim=dim, dst_top_k_element=k, dst_top_k_dim=dim)
    lossy_dense, sst, dst = sparser.lossy_compress(tensor.to(device))
    objects_are_equal(sst.to(device), expd_sst.to(device), raise_exception=True, rtol=RTOL, atol=ATOL)
    objects_are_equal(dst.to(device), expd_dst.to(device), raise_exception=True, rtol=RTOL, atol=ATOL)
    objects_are_equal(lossy_dense.to(device), expd_rt.to(device), raise_exception=True, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "tensor, dim, top_k_percent",
    [
        (torch.linspace(0.01, 0.06, 40).reshape(5, 8), 0, 100),
        (torch.linspace(-0.01, 0.06, 42).reshape(7, 6), 0, 100),
        (torch.linspace(-10, 15, 36).reshape(6, 6), 1, 100),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lossy_compress_sparsity_0(tensor, dim, top_k_percent, device):
    """Tests whether lossy_compress method simply returns dense tensor when sparsity is 0."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")

    sparser = SignalSparsity(
        sst_top_k_percent=top_k_percent, sst_top_k_dim=dim, dst_top_k_percent=top_k_percent, dst_top_k_dim=dim
    )
    lossy_dense, sst, dst = sparser.lossy_compress(tensor.to(device))
    objects_are_equal(lossy_dense.to(device), tensor.to(device), raise_exception=True, rtol=RTOL, atol=ATOL)
    objects_are_equal(sst, None, raise_exception=True, rtol=RTOL, atol=ATOL)
    objects_are_equal(dst.to(device), tensor.to(device), raise_exception=True, rtol=RTOL, atol=ATOL)


def test_sst_disabled():
    """Tests the case where SST is disabled."""
    dense = torch.tensor([0.5000, 0.6000, 0.7000, 0.8000])
    result = torch.tensor([0.0, 0.0, 0.7000, 0.8000])
    sparser = SignalSparsity(dst_top_k_element=2, dst_top_k_dim=0)
    rt, sst, dst = sparser.lossy_compress(dense)
    objects_are_equal(rt, result, raise_exception=True, rtol=RTOL, atol=ATOL)
    objects_are_equal(dst, result, raise_exception=True, rtol=RTOL, atol=ATOL)
    assert sst is None


def test_dst_disabled():
    """Tests the case where DST is disabled."""
    dense = torch.tensor([0.5000, 0.6000, 0.7000, 0.8000, 0.9000])
    result_rt = torch.tensor([0.6000, 0.7618, 0.7000, 0.6382, 0.8000])
    result_sst = torch.tensor(
        [
            3.50000000000000000000 + 0.00000000000000000000j,
            0.00000000000000000000 + 0.00000000000000000000j,
            -0.25000002980232238770 + 0.08122986555099487305j,
            -0.25000002980232238770 - 0.08122986555099487305j,
            0.00000000000000000000 + 0.00000000000000000000j,
        ]
    )
    sparser = SignalSparsity(sst_top_k_element=3, sst_top_k_dim=0)
    rt, sst, dst = sparser.lossy_compress(dense)
    objects_are_equal(rt, result_rt, raise_exception=True, rtol=RTOL, atol=ATOL)
    objects_are_equal(sst, result_sst, raise_exception=True, rtol=RTOL, atol=ATOL)
    assert dst is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_sparse_mask(device):
    """Tests random_sparse_mask API."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")

    dense = torch.tensor([0.5000, 0.6000, 0.7000, 0.8000, 0.9000]).to(device)
    mask = random_sparse_mask(dense, 20, 0)
    assert mask.sum() == 1
    for d in [0, 1]:
        dense = torch.rand(100, 100).to(device)
        mask = random_sparse_mask(dense, 1, d)
        assert objects_are_equal(mask.sum(dim=d), torch.ones(100).to(device), raise_exception=True)
        assert mask.sum() == 100
