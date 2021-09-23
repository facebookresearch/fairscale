# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from fairscale.nn.misc import ParamBucket


def test_param_values_conserved():
    param = torch.rand((2, 3))

    bucket = ParamBucket(10, param.dtype, param.device)
    param_ = param.clone()

    bucket.add_param(param_)
    torch.allclose(param, param_)


def test_max_size():
    param = torch.rand((20, 30))

    bucket = ParamBucket(5, param.dtype, param.device)
    with pytest.raises(AssertionError):
        bucket.add_param(param)


def test_double_check_int():
    param = torch.rand((5, 6))

    bucket = ParamBucket(300, param.dtype, param.device)
    bucket.add_param(param)

    with pytest.raises(AssertionError):
        bucket.add_param(param)


def test_type_change():
    size = (5, 6)
    param = torch.rand(size, requires_grad=True)
    param_ = param.clone()

    bucket = ParamBucket(30, param.dtype, param.device)
    bucket.add_param(param)

    # Move the bucket to fp16 and back
    bucket.to(dtype=torch.float16, device=param.device)
    assert bucket.buffer.dtype == torch.float16

    bucket.to(dtype=torch.float32, device=param.device, keep_param_alignment=True)
    assert bucket.buffer.dtype == torch.float32

    # Same with the reference tensor
    param_.to(dtype=torch.float16)
    param_.to(dtype=torch.float32)

    torch.allclose(param, param_)
