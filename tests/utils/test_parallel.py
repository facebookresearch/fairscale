# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test utility classes from fairscale.utils.parallel """

from parameterized import parameterized
import torch

from fairscale.internal.parallel import chunk_and_pad


@parameterized.expand([[num_chunks] for num_chunks in range(1, 33)])
def test_chunk_and_pad(num_chunks):
    max_tensor_size = 256
    tensor = torch.zeros(max_tensor_size)
    for tensor_size in range(1, max_tensor_size + 1):
        tensor_i = tensor[:tensor_size]
        chunks = chunk_and_pad(tensor_i, num_chunks)
        assert len(chunks) == num_chunks
        assert all(len(chunks[0]) == len(chunk) for chunk in chunks)
