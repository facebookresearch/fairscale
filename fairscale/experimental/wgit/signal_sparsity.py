# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional

import torch
from torch import Tensor


# Helper Functions
def _get_k_for_topk(topk_percent: Optional[float], top_k_element: Optional[int], top_k_total_size: int) -> int:
    """Converts the top_k_percent to top_k_element when top_k_percent is provided
    as the criterion for top-k calculation. When, top_k_element is used as the criterion,
    simply returns the value for k. Also, ensures k is never 0 to avoid all-zero tensors.
    """
    if top_k_element is None:
        top_k_element = int(top_k_total_size * topk_percent / 100.0)
    elif top_k_element > top_k_total_size:
        raise ValueError("top_k_element for sst or dst is larger than max number of elements along top_k_dim")
    # ensure we never have 100% sparsity in tensor and always have 1 surviving element!
    return max(1, top_k_element)


def _scatter_topk_to_sparse_tensor(
    top_k_tensor: Tensor, to_be_sparsify_tensor: Tensor, k: int, dim: Optional[int]
) -> Tensor:
    """Scatter the topk values of the to_be_sparsify_tensor to a zero tensor of the same shape
    at the top-k indices of the top_k_tensor. This function allows top-k computation with a
    derived tensor from to_be_sparsify_tensor.

    Args:
        top_k_tensor (Tensor):
            The source tensor whose top-k "indices" are taken and used to extract
            the corresponding "values" from the to_be_sparsify_tensor.
        to_be_sparsify_tensor (Tensor):
            The tensor whose values are gathered according to the top-k indices
            of the top_k_tensor, and a zero tensor of same shape is populated with these
            values at those indices and creates the sparse_tensor tensor.
        k (int):
            the value of k for top-k
        dim (Optional[int]):
            dimension for top-k

    Returns:
        (Tensor):
            Returns a sparse_tensor with the same shape as the top_k_tensor and to_be_sparsify_tensor,
            and populated with the values of the to_be_sparsify_tensor at the indices corresponding
            to the top-k indices of the source tensor.
    """
    assert (
        top_k_tensor.shape == to_be_sparsify_tensor.shape
    ), "top_k_tensor and to_be_sparsify_tensor have different shapes!"

    sparse_tensor = torch.zeros_like(to_be_sparsify_tensor)
    orig_shape = sparse_tensor.shape
    if dim is None and len(orig_shape) > 1:
        sparse_tensor = sparse_tensor.reshape(-1)
        to_be_sparsify_tensor = to_be_sparsify_tensor.reshape(-1)
        top_k_tensor = top_k_tensor.reshape(-1)
        dim = -1

    _, i = top_k_tensor.topk(k, dim=dim)
    return sparse_tensor.scatter(dim, i, to_be_sparsify_tensor.gather(dim, i)).reshape(orig_shape)


def _top_k_total_size(tensor: Tensor, topk_dim: Optional[int]) -> int:
    """Get the total size of the input tensor along the topk_dim dimension. When, the
    dimension is None, get the number of elements in the tensor.
    """
    top_k_total_size = tensor.numel() if topk_dim is None else tensor.shape[topk_dim]
    assert top_k_total_size > 0, "Total size of input tensor along the topk_dim has to be greater than 0."
    return top_k_total_size


def _dct_transform(dense: Tensor) -> Tensor:
    """Should take a tensor and perform a Discrete Cosine Transform on the tensor.

    Args:
        dense (Tensor):
            Input dense tensor (no zeros).
    Returns:
        (Tensor):
            transformed dense tensor DCT components
    """
    raise NotImplementedError("Support for DCT has not been implemented yet!")


def _inverse_dct_transform(dense: Tensor) -> Tensor:
    """Should take a tensor and perform an inverse Discrete Cosine Transform on the tensor.

    Args:
        dense (Tensor):
            Input dense tensor (no zeros).
    Returns:
        (Tensor):
            transformed dense tensor with iDCT values
    """
    raise NotImplementedError("Support for iDCT has not been implemented yet!")


class Algo(Enum):
    FFT = 0
    DCT = 1


class SignalSparsity:
    """
    This class represents a particular config for a set of signal
    processing based sparsification functions on tensors. This can
    be used both on weights, gradients and other tensors like the
    optimizer state.

    During initialization, this class requires a value for one of
    `sst_top_k_element` or `sst_top_k_percent` and also requires a
    value for one of `dst_top_k_element` or `dst_top_k_percent`.

    Args:
        algo (Algo):
            The algorithm used. Default: FFT
        sst_top_k_dim (int, optional):
            The dimension on which the top-k is done for SST.
            E.g. -1 is the last dim. None means flatten and top-k on all dims.
            There is no way to specify multiple dims other than None.
            Default: -1
        sst_top_k_element (int, optional):
            Number of top-k elements to retain for SST. Default: None
        sst_top_k_percent (float, optional):
            Percent of top-k elements to retain for SST. Default: None
        dst_top_k_dim (int, optional):
            The dimension on which the top-k is done for DST.
            E.g. -1 is the last dim. None means flatten and top-k on all dims.
            There is no way to specify multiple dims other than None.
            Default: None
        dst_top_k_element (int, optional):
            Number of top-k elements to retain for DST. Default: None
        dst_top_k_percent (float, optional):
            Percent of top-k elements to retain for DST. Default: None

    Example:
        .. code-block:: python

            2d_sparser = SignalSparsity()
            sst, dst = 2d_sparser.get_sst_dst(linear.weight.data)

            3d_sparser = SingalSparsity(algo=Algo.DCT, sst_top_k_dim=None, dst_top_k_dim=-1, sst_top_k_percent=10, dst_top_k_element=100)
            conv.weight.data = 3d_sparser.get_sst_dst_weight(conv.weight.data)
    """

    def __init__(
        self,
        algo: Algo = Algo.FFT,
        sst_top_k_dim: Optional[int] = -1,
        sst_top_k_element: Optional[int] = None,
        sst_top_k_percent: Optional[float] = None,
        dst_top_k_dim: Optional[int] = -1,
        dst_top_k_element: Optional[int] = None,
        dst_top_k_percent: Optional[float] = None,
    ) -> None:

        self._sst_top_k_dim = sst_top_k_dim
        self._sst_top_k_element = sst_top_k_element
        self._sst_top_k_percent = sst_top_k_percent
        self._dst_top_k_dim = dst_top_k_dim
        self._dst_top_k_element = dst_top_k_element
        self._dst_top_k_percent = dst_top_k_percent

        self._validate_conf()
        # TODO (Min): Type checking for the following
        self._transform = torch.fft.fft if algo is Algo.FFT else _dct_transform  # type: ignore
        self._inverse_transform = torch.fft.ifft if algo is Algo.FFT else _inverse_dct_transform  # type: ignore

    def _validate_conf(self) -> None:
        """Validating if the config is valid.

        This includes asserting the following:
        1. validating that one and only one of top_k_element and top_k_percent is set.
        2. Asserting that both element and percentage are in valid ranges.

        Throws:
            ValueError:
                If validation fails.
        """
        # assert that both top_k_elements and top_k_percent aren't set for sst and dst
        def one_and_only(a: Optional[int], b: Optional[float]) -> bool:
            return (a is None) ^ (b is None)

        if not (
            one_and_only(self._sst_top_k_element, self._sst_top_k_percent)
            and one_and_only(self._dst_top_k_element, self._dst_top_k_percent)
        ):
            raise ValueError(
                "One and only one of top_k_element and top_k_percent for "
                "each of sst and dst must be provided as an argument.\n"
                f"Input values are: sst element={self._sst_top_k_element}, sst percent={self._sst_top_k_percent}, "
                f"dst element={self._dst_top_k_element}, dst percent={self._dst_top_k_percent}"
            )

        # assert that, if top_k_percent is not None, it is a valid number for a percentage.
        def none_or_in_range(a: Optional[float]) -> bool:
            return a is None or (0.0 < a <= 100.0)

        if not (none_or_in_range(self._sst_top_k_percent) and none_or_in_range(self._dst_top_k_percent)):
            raise ValueError(
                "top_k_percent values for sst and dst has to be in the interval (0, 100].\n"
                f"Input values are: sst percent={self._sst_top_k_percent}, dst percent={self._dst_top_k_percent}"
            )

        def none_or_greater_0(a: Optional[int]) -> bool:
            return a is None or (0 < a)

        if not (none_or_greater_0(self._sst_top_k_element) and none_or_greater_0(self._dst_top_k_element)):
            raise ValueError(
                "top_k_element values for sst and dst has to be greater than 0.\n"
                f"Input values are: sst element={self._sst_top_k_element} "
                f"and dst element={self._dst_top_k_element}"
            )

    def dense_to_sst(self, dense: Tensor) -> Tensor:
        """Get Signal Sparse Tensor (SST) from a dense tensor

        Dense -> fft -> top-k -> results.

        The input dense tensor is transformed using a transform algorithm according to the `algo`
        initialization argument. The SST is then generated from the top_k_elements
        (or the top_k_percentage) of values from the transformed tensor along the 'sst_top_k_dim'.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).

        Returns:
            (Tensor):
                Same shaped tensor as the input dense tensor, still in dense format but in frequency
                domain (complex valued) and has zeros.
        """
        top_k_total_size = _top_k_total_size(dense, self._sst_top_k_dim)
        k = _get_k_for_topk(self._sst_top_k_percent, self._sst_top_k_element, top_k_total_size)
        dense_freq = self._transform(dense)

        # NOTE: real_dense_freq can potentially be magnitude of complex frequency components
        # or DCT transformed components when using DCT (currently not implemented).
        # TODO: In case of the FFT, the imaginary part can perhaps be quantized or pruning can be
        # done on the smaller phases.
        real_dense_freq = dense_freq.real.abs()
        return _scatter_topk_to_sparse_tensor(real_dense_freq, dense_freq, k, dim=self._sst_top_k_dim)

    def dense_sst_to_dst(self, dense: Tensor, sst: Tensor) -> Tensor:
        """Calculates DST from input dense and SST tensors.

        dense - inverse_transform(sst)[using sst_dst_to_dense method] -> top-k -> dst

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).
            sst (Tensor):
                Input SST tensor (has zeros).

        Returns:
            (Tensor):
                Same shaped tensor, still dense format but has zeros. Non-zeros are top-k delta values.
        """
        delta = dense - self.sst_dst_to_dense(sst)  # sst_dst_to_dense(sst) returns the inverse transform here
        top_k_total_size = _top_k_total_size(dense, self._dst_top_k_dim)
        k = _get_k_for_topk(self._dst_top_k_percent, self._dst_top_k_element, top_k_total_size)
        return _scatter_topk_to_sparse_tensor(delta.abs(), delta, k, dim=self._dst_top_k_dim)

    def sst_dst_to_dense(self, sst: Tensor, dst: Optional[Tensor] = None) -> Tensor:
        """From SST and DST returns a dense reconstruction. When argument dst=None, simply returns
        the inverse transform of the SST tensor.

        Args:
            sst (Tensor):
                Singal sparse tensor. Required argument.
            dst (Tensor, optional):
                Delta sparse tensor, optional.

        Returns:
            (Tensor):
                A dense tensor in real number domain from the SST.
        """
        dense_recons = self._inverse_transform(sst).real
        if dst is not None:
            dense_recons += dst
        return dense_recons


# We could separate have helper functions that work on state_dict instead of a tensor.
# One option is to extend the above class to handle state_dict as well as tensor
# but we may want to filter on the keys in the state_dict, so maybe that option isn't
# the best. We need to have further discussions on this.
