# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing SSD Offload Module
"""

import filecmp
import os
import tempfile

import numpy as np
import torch

import fairscale.experimental.nn.ssd_offload as so


def _init():
    torch.manual_seed(0)
    np.random.seed(0)


def test_write_read():
    _init()

    with tempfile.NamedTemporaryFile() as f:
        ref_tensor = torch.rand((128), dtype=torch.float32)
        test_tensor = torch.zeros_like(ref_tensor)
        assert not torch.equal(ref_tensor, test_tensor)
        so.write(ref_tensor, f.name)
        so.read(test_tensor, f.name)
        assert torch.equal(ref_tensor, test_tensor)


def test_torch_save_load():
    _init()
    orig_file = tempfile.NamedTemporaryFile()
    checkpoint_file = tempfile.NamedTemporaryFile()

    # TENSOR_SHAPE = (1024, 1024, 1024)
    # use smaller shape for unit tests
    TENSOR_SHAPE = (1024, 1024)
    ref_tensor = torch.rand(TENSOR_SHAPE, dtype=torch.float32)
    ref_ssd_tensor = so.SsdTensor.fromtensor(ref_tensor, orig_file.name)
    del ref_tensor
    # after deleting ref_tensor, memory usage should be very low
    # For save it shouldn't be more than 10x so.DEFAULT_CHUNK_SIZE
    so.torch_saver.save(ref_ssd_tensor, checkpoint_file.name)
    # below line saves file to orig_file.name+"_2"
    # Memory usage here should be O(1000 * so.DEFAULT_CHUNK_SIZE)
    # 1000x because that's how many elements the python unpickler
    # will buffer before passing to the SsdTensor
    test_ssd_tensor = torch.load(checkpoint_file)
    assert filecmp.cmp(orig_file.name, orig_file.name + "_2", shallow=False)
    os.unlink(orig_file.name + "_2")
