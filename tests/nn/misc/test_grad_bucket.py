# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from fairscale.nn.misc import GradBucket


def test_grad_values_conserved():
    with torch.no_grad():  # remove a warning
        param = torch.rand((2, 3), requires_grad=True)
        param.grad = torch.rand(2, 3)

        bucket = GradBucket(10, param.dtype, param.device, -1)
        param_ = param.clone()

        bucket.add_grad(param_)
        torch.allclose(param.grad, param_.grad)


def test_memory_leak():
    with torch.no_grad():  # remove a warning
        param = torch.rand((2, 3), requires_grad=True)
        param.grad = torch.rand(2, 3)

        bucket = GradBucket(300, param.dtype, param.device, -1)
        bucket.add_grad(param)
        bucket.shrink()

        storage = bucket.buffer.storage()
        # See https://github.com/pytorch/pytorch/pull/59671/
        if hasattr(storage, "nbytes"):
            assert storage.nbytes() == 6 * bucket.buffer.element_size()
        else:
            assert len(storage) == 6


def test_max_size():
    with torch.no_grad():  # remove a warning
        param = torch.rand((20, 30), requires_grad=True)
        param.grad = torch.rand(20, 30)

        bucket = GradBucket(5, param.dtype, param.device, -1)
        with pytest.raises(AssertionError):
            bucket.add_grad(param)


def test_collapse():
    with torch.no_grad():  # remove a warning
        size = (5, 6)
        param = torch.rand(size, requires_grad=True)
        param.grad = torch.rand(size)

        bucket = GradBucket(300, param.dtype, param.device, -1)
        bucket.add_grad(param)
        bucket.shrink()
        bucket.collapse()

        assert bucket.buffer.numel() == 0
        assert param.grad is None
        bucket.rebuild()

        assert param.grad is not None
        torch.allclose(param.grad, torch.zeros(size))
