# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch


class ParamBucket:
    """
    Helper class to simplify the handling of parameter buckets
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) -> None:
        self._params: List[torch.Tensor] = []
        self._param_ids: List[int] = []
        self._fill = 0

        # The actual flat tensor
        self.buffer: torch.Tensor = torch.zeros(size, dtype=dtype, device=device)
        self.sent = True

    def to(  # type: ignore
        self,
        device: Optional[Union[int, torch.device]],
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        keep_param_alignment: bool = True,
    ) -> "ParamBucket":
        """
        Move the underlying buffer
        """
        assert self.buffer is not None, "Cannot move a collapsed bucket, please rebuild it"
        self.buffer.to(device, dtype, non_blocking)

        if keep_param_alignment:
            self._reattach_params()

    @torch.no_grad()
    def add_param(self, param: torch.Tensor) -> None:
        """
        Add a new parameter gradient to the bucket. Param.grad becomes a view of this bucket buffer
        """

        assert id(param) not in self._param_ids, "The same param cannot be checked in twice"

        self._add_param_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @torch.no_grad()
    def _add_param_as_view(self, param: torch.Tensor, keep_existing_value: bool = True) -> None:
        assert self.buffer is not None
        assert param.dtype == self.buffer.dtype
        assert param.device == self.buffer.device

        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()

        # Copy the current param value
        if keep_existing_value:
            self.buffer[self._fill : fill_next].copy_(param.data.flatten())
        param.data = self.buffer[self._fill : fill_next].view_as(param.data)
        self._fill = fill_next

    @torch.no_grad()
    def _reattach_params(self) -> None:
        """
        Given the parameters which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0

        self._fill = 0
        for p in self._params:
            self._add_param_as_view(p, keep_existing_value=False)
