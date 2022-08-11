# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import Tensor


class EnergyConcentrationProfile:
    """Compute "energy" concentration level for a tensor

    Args:
        dim (int):
            The dimension to measure.
        top_k_percents (List[float]):
            List of percentage values. For each value, the `measure`
            function will compute and return the percentage of "energy"
            concentrated on that top-K percent of values in the dimension
            to measure. Note, this is the opposite of the sparsity percentage.
    """

    def __init__(self, dim: int, top_k_percents: List[float]) -> None:
        assert isinstance(dim, int)
        self.dim = dim
        self.percents = []
        last_p = 0.0
        for p in top_k_percents:
            assert isinstance(p, (int, float))
            assert p > 0, p
            assert p <= 100, p
            assert p > last_p, f"p {p} should be larger than last_p {last_p}"
            self.percents.append(float(p))
            last_p = p

    def measure(self, in_tensor: Tensor) -> List[Tensor]:
        """Compute the return the results

            Note, we want this function to be nonblocking and async.

        Returns:
            (List[Tensor])
                List of tensors. Each tensor is a singleton float
                that contains the energy measure for that top_k_percent.
        """
        assert in_tensor.is_floating_point(), in_tensor.dtype
        assert self.dim < len(in_tensor.shape), f"tensor shape {in_tensor.shape} not compatible with dim {self.dim}"
        dim_size = in_tensor.shape[self.dim]
        abs_tensor = in_tensor.abs()
        full_energy = abs_tensor.sum()
        return_tensors = []
        for p in self.percents:
            k = max(1, round(p / 100 * dim_size))
            abs_top_k_values, _ = abs_tensor.topk(k, dim=self.dim)
            return_tensors.append(abs_top_k_values.sum() / full_energy)
        return return_tensors

    def measure_fft(self, in_tensor: Tensor) -> List[Tensor]:
        """Like measure, but do it in FFT frequency domain."""
        return self.measure(torch.fft.fft(in_tensor, dim=self.dim).real)
