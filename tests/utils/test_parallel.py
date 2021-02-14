# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test utility classes from containers.py. """

import pytest
import torch

from fairscale.utils.parallel import compute_shard_size


@pytest.mark.parametrize(
    "test_case", [(1, 2), (2, 2), (3, 2), (4, 2), (4, 4), (3, 4), (9, 4), (9, 6), (10, 5), (14, 5)]
)
def test_compute_shard_size(test_case):
    """Test compute_shard_size, verify using torch.chunk()"""
    numel, world_size = test_case
    result = compute_shard_size(numel, world_size)
    expected = torch.zeros(numel).chunk(world_size)[0].numel()
    assert result == expected, f"{result} == {expected}"
