# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from ... import Tensor
from .. import Parameter
from .module import Module
from typing import Any, Optional


class _BatchNorm(Module):
    num_features: int = ...
    eps: float = ...
    momentum: float = ...
    affine: bool = ...
    track_running_stats: bool = ...
    weight: Parameter = ...
    bias: Parameter = ...

    # This field is used by fairscale.nn.misc.misc::patch_batchnorm
    _track_running_stats_backup: bool

#MODIFIED BY TORCHGPIPE
    running_mean: Tensor
    running_var: Tensor
    num_batches_tracked: Tensor

    def __init__(self, num_features: int, eps: float = ..., momentum: Optional[float] = ..., affine: bool = ...,
                 track_running_stats: bool = ...) -> None: ...
#END

    def reset_running_stats(self) -> None: ...

    def reset_parameters(self) -> None: ...


class BatchNorm1d(_BatchNorm): ...


class BatchNorm2d(_BatchNorm): ...


class BatchNorm3d(_BatchNorm): ...


class SyncBatchNorm(_BatchNorm):
    # TODO set process_group to the write type once torch.distributed is stubbed
    def __init__(self, num_features: int, eps: float = ..., momentum: float = ..., affine: bool = ...,
                 track_running_stats: bool = ..., process_group: Optional[Any] = ...) -> None: ...
