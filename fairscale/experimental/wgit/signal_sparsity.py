# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

from torch import Tensor


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

    def __init__(self) -> None:
        pass
        self._validate_conf()

    def _validate_conf(self) -> None:
        """Validating the config is valid.

        For example, not both top_k_element and top_k_percent is set.

        this should assert fail if checking fails.
        """
        pass

    def dense_to_sst(self, dense: Tensor) -> Tensor:
        """Get SST from a tensor

        Dense -> fft -> top-k -> results.

        Returns:
            Same shaped tensor, still in dense format but in frequency domain and has zeros.
        """
        pass

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


# We could separate have helper functions that work on state_dict instead of a tensor.
# One option is to extend the above class to handle state_dict as well as tensor
# but we may want to filter on the keys in the state_dict, so maybe that option isn't
# the best. We need to have further discussions on this.
