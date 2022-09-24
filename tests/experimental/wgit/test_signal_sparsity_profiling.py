# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import time

import pytest
import torch

from fairscale.experimental.wgit.signal_sparsity_profiling import EnergyConcentrationProfile as ECP
from fairscale.fair_dev.testing.testing import objects_are_equal, skip_if_no_cuda

# Our own tolerance
ATOL = 1e-6
RTOL = 1e-5

# enable this for debugging.
# torch.set_printoptions(precision=20)


@skip_if_no_cuda
def test_nonblocking():
    """Tests cpu runs ahead of the GPU in the measuring process."""
    big = torch.rand(10, 1000, 1000).cuda()
    ecp = ECP(dim=2, top_k_percents=[1, 5, 10, 50, 90])
    start = time.time()
    out = ecp.measure(big)
    out_fft = ecp.measure_fft(big)
    cpu_time = time.time() - start
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    assert cpu_time * 5 < gpu_time, f"GPU time should dominate {cpu_time} vs. {gpu_time}"
    for o in [out, out_fft]:
        # validate the output
        p = [x.item() for x in o]
        for n, n1 in zip(p, p[1:]):
            assert n <= n1 and n >= 0 and n <= 100, f"n={n} n1={n1}"


def get_ones():
    """Return test data with ones tensor"""
    return (
        0,
        [1, 5, 10, 100],
        torch.ones(100),
        [torch.tensor(0.01), torch.tensor(0.05), torch.tensor(0.1), torch.tensor(1.0)],
    )


def get_dim_0():
    """Test case for dim=0 for 2D input."""
    return (
        0,
        [1, 3, 33, 66, 90],
        torch.tensor([0.1, 0.2, 0.1, 0.45]).repeat(100, 1),
        [torch.tensor(0.01), torch.tensor(0.03), torch.tensor(0.33), torch.tensor(0.66), torch.tensor(0.9)],
    )


@pytest.mark.parametrize(
    "dim, percents, in_tensor, out_tensors",
    [
        get_ones(),
        get_dim_0(),
    ],
)
def test_expected_output(dim, percents, in_tensor, out_tensors):
    """Test with a few expected input & outputs."""
    ecp = ECP(dim, percents)
    out = ecp.measure(in_tensor)
    objects_are_equal(out, out_tensors, raise_exception=True, rtol=RTOL, atol=ATOL)
    out_fft = ecp.measure_fft(torch.fft.ifft(in_tensor, dim=dim))
    objects_are_equal(out_fft, out_tensors, raise_exception=True, rtol=RTOL, atol=ATOL)
