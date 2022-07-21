# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional

import torch
from torch import Tensor


# Helper Functions
def get_k_for_topk(
    tensor: Tensor, topk_percent: Optional[float], top_k_element: Optional[int], topk_dim: Optional[int]
) -> int:
    """Converts the top_k_percent to top_k_element when top_k_percent is provided
    as the criterion for top-k calculation. When, top_k_element is used as the criterion,
    simply returns the value for k. Also, ensures k is never 0 to avoid all sparse tensors.

    Args:
        tensor (Tensor):
            Input tensor.
        topk_percent (float, optional)
            Percent of top-k elements to retain for SST. Default: 0.1

    Returns:
        k (int)
            the value of k for use as top-k
    """
    if top_k_element is None:
        dim_numel = tensor.numel() if topk_dim is None else tensor.shape[topk_dim]
        k = int(dim_numel * topk_percent / 100.0)
    else:
        k = top_k_element

    # ensure we never have 100% sparsity in tensor and always have 1 surviving element!
    k = k if k != 0 else 1
    return k


def scatter_topk_to_sparse_tensor(src_tensor: Tensor, dest_tensor: Tensor, k: int, dim: Optional[int]) -> Tensor:
    """Scatter the topk values of the dest_tensor to a zero tensor of the same shape
    at the top-k indices of the src_tensor.

    Args:
        src_tensor (Tensor)
            The source tensor whose top-k "indices" are taken and used to extract
            the corresponding "values" from the dest_tensor.
        dest_tensor (Tensor)
            The tensor whose values are gathered according to the top-k indices
            of the src_tensor, and a zero tensor of same shape is populated with these
            values at those indices and creates the sparse_dest tensor.

    Returns:
        Returns a sparse_dest tensor with the values of the dest_tensor corresponding
        to the top-k indices of the source tensor.

    """
    _, idx = src_tensor.topk(k, dim=dim)
    sparse_dest = torch.zeros_like(dest_tensor)
    # gather the values from the dest_tensor and scatter to a tensor of zeros
    sparse_dest = sparse_dest.scatter_(dim, idx, dest_tensor.gather(dim, idx))
    return sparse_dest


class Algo(Enum):
    FFT = 0
    DCT = 1


class SignalSparsity:
    """
    This class represents a particular config for a set of signal
    processing based sparsification functions on tensors. This can
    be used both on weights, gradients and other tensors like the
    optimizer state.

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
            Percent of top-k elements to retain for SST. Default: 0.1
        dst_top_k_dim (int, optional):
            The dimension on which the top-k is done for DST.
            E.g. -1 is the last dim. None means flatten and top-k on all dims.
            There is no way to specify multiple dims other than None.
            Default: None
        dst_top_k_element (int, optional):
            Number of top-k elements to retain for DST. Default: None
        dst_top_k_percent (float, optional):
            Percent of top-k elements to retain for DST. Default: 0.1

    Example:
        .. code-block:: python

            2d_sparser = SignalSparsity()
            sst, dst = 2d_sparser.get_sst_dst(linear.weight.data)

            3d_sparser = SingalSparsity(algo=Algo.DCT, sst_top_k_dim=None, dst_top_k_dim=-1, dst_top_k_element=5, dst_top_k_percent=None)
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
        self._transform = torch.fft.fft if algo == Algo.FFT else self._dct_transform  # type: ignore

    def _validate_conf(self) -> None:
        """Validating the config is valid.

        This includes asserting the following:
        1. validating that both top_k_element and top_k_percent is not set.
        2. Asserting that if top_k_percent is not None, it is a valid number for percentage.

        This asserts fail if the checking fails.
        """
        # assert that both top_k_elements and top_k_percent aren't set for sst and dst
        assert (self._sst_top_k_element is None) ^ (
            self._sst_top_k_percent is None
        ), "One and only one of sst_top_k_element and sst_top_k_percent has to be provided as argument"
        assert (self._dst_top_k_element is None) ^ (
            self._dst_top_k_percent is None
        ), "One and only one of dst_top_k_element and dst_top_k_percent has to be provided as argument"

        # assert, if top_k_percent is not None, it is a valid number for percentage.
        assert self._sst_top_k_percent is None or (
            0.0 < self._sst_top_k_percent <= 100.0
        ), "The value for the argument sst_top_k_percent has to be in the interval (0, 100]"

        assert self._dst_top_k_percent is None or (
            0.0 < self._dst_top_k_percent <= 100.0
        ), "The value for the argument dst_top_k_percent has to be in the interval (0, 100]"

    def dense_to_sst(self, dense: Tensor) -> Tensor:
        """Get SST from a tensor

        Dense -> fft -> top-k -> results.

        Generates a Signal Sparse Tensor (SST) from an input dense tensor. The input dense
        tensor is transformed using a transform algorithm according to the `algo` initialization
        argument. The SST is then generated from the top_k_elements (or the top_k_percentage) of
        values from the transformed tensor along the 'sst_top_k_dim'.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).

        Returns:
            Same shaped tensor, still in dense format but in frequency domain and has zeros.
        """
        k = get_k_for_topk(dense, self._sst_top_k_percent, self._sst_top_k_element, self._sst_top_k_dim)
        dense_freq = self._transform(dense)

        # NOTE: real_dense_freq can potentially be magnitude of complex frequency components
        # or DCT transformed components when using DCT (currently not implemented).
        # TODO: In case of the FFT, the imaginary part can perhaps be quantized or pruning can be
        # done on the smaller phases.
        real_dense_freq = dense_freq.real

        sst = scatter_topk_to_sparse_tensor(real_dense_freq, dense_freq, k, dim=self._sst_top_k_dim)
        return sst

    def dense_sst_to_dst(self, dense: Tensor, sst: Tensor) -> Tensor:
        """From dense and SST to a DST

        This will use sst_dst_to_dense below but with dst=None.

        dense - ifft(sst)[using sst_dst_to_dense below) -> top-k -> result

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).
            sst (Tensor):
                Input SST tensor (has zeros).

        Returns:
            Same shaped tensor, still dense format but has zeros. Non-zeros are top-k delta values.
        """
        pass

    def sst_dst_to_dense(self, sst: Tensor, dst: Tensor = None) -> Tensor:
        """From SST and dst back to a dense

        result = ifft(sst)
        if dst is not None:
            result += dst
        return result

        Args:
            sst (Tensor):
                Singal sparse tensor. Required argument.
            dst (Tensor, optinoal):
                Delta sparse tensor, optional.

        Returns:
            A dense tensor in real number domain from the SST.
        """
        pass

    def sst_or_dst_to_mask(self) -> None:
        # we shouldn't need this function since going from SST/DST to mask should be a
        # trivial call in pytorch. Maybe I am missing something.
        pass

    def _dct_transform(self, dense: Tensor) -> Tensor:
        """Should take a tensor and perform a Discrete Cosine Transform on the tensor.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).
        Returns:
            transformed dense tensor DCT components
        """
        raise NotImplementedError()


# We could separate have helper functions that work on state_dict instead of a tensor.
# One option is to extend the above class to handle state_dict as well as tensor
# but we may want to filter on the keys in the state_dict, so maybe that option isn't
# the best. We need to have further discussions on this.
