# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing SsdBuffer and SsdTensorHandle modules.
"""

import filecmp
import os
import tempfile

import numpy as np
import pytest
import torch

import fairscale.experimental.nn.ssd_offload as so
from fairscale.utils import torch_version

# Note: We need the nightly version for SSD offload to work. Hence I am checking for the next PyTorch release.
pytestmark = pytest.mark.skipif(torch_version() < (1, 11, 0), reason="requires torch version >= 1.11.0")


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


def test_ssd_handle_dispatch_fwd():
    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((128))
        ssd_handle = so.SsdTensorHandle.from_tensor(orig_tensor)
        ssd_handle.set_file_params(f.name, 0)
        ssd_handle.to_file(release_tensor_after_write=True)

        assert torch.equal(ssd_handle.to_tensor(), orig_tensor)

        # This should trigger the torch_dispatch code and write
        # back the results to the file
        ssd_handle.add_(1)
        plus1_tensor = orig_tensor.add(1)
        assert torch.equal(ssd_handle.to_tensor(), plus1_tensor)


def test_ssd_handle_dispatch_bwd():
    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((4, 4), requires_grad=True)
        orig_copy = orig_tensor.clone().detach().requires_grad_(True)
        ssd_handle = so.SsdTensorHandle.from_tensor(orig_tensor)
        ssd_handle.set_file_params(f.name, 0)
        ssd_handle.to_file(release_tensor_after_write=True)

        assert torch.equal(ssd_handle.to_tensor(), orig_tensor)

        y1 = ssd_handle + 1
        y2 = orig_copy + 1
        y1.sum().backward()
        y2.sum().backward()

        # TODO: PJ/ASenable assert once Tensor._make_subclass can properly define the tensor's shape
        # assert torch.equal(ssd_handle.grad, orig_copy.grad)


def test_ssd_buffer_basic():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_tensor = torch.rand((128), dtype=torch.float32)
        refb_tensor = torch.rand((128), dtype=torch.float32)
        refc_tensor = torch.rand((128), dtype=torch.float32)
        ssd_buf = so.SsdBuffer(1024, f.name)

        hdl_a = ssd_buf.insert(refa_tensor)
        hdl_b = ssd_buf.insert(refb_tensor)
        hdl_c = ssd_buf.insert(refc_tensor)

        assert hdl_a.is_available()
        assert hdl_b.is_available()
        assert hdl_c.is_available()

        assert torch.equal(refa_tensor, hdl_a.get_tensor())
        assert torch.equal(refb_tensor, hdl_b.get_tensor())
        assert torch.equal(refc_tensor, hdl_c.get_tensor())

        tensors = ssd_buf.get_tensors()
        assert hdl_a is tensors[0]
        assert hdl_b is tensors[1]
        assert hdl_c is tensors[2]

        # test read_into_tensor when handle.is_available()
        b_tensor_copy1 = torch.empty_like(refb_tensor)
        hdl_b.copy_into_tensor(b_tensor_copy1)
        assert torch.equal(refb_tensor, b_tensor_copy1)

        # remove references so memory will be cleaned up
        buffer = None

        ssd_buf.to_disk()

        assert hdl_a.filename == f.name
        assert hdl_b.filename == f.name
        assert hdl_c.filename == f.name

        assert hdl_a.offset == 0
        assert hdl_b.offset == 128
        assert hdl_c.offset == 256

        assert not hdl_a.is_available()
        assert not hdl_b.is_available()
        assert not hdl_c.is_available()

        # test read_into_tensor when !handle.is_available()
        b_tensor_copy2 = torch.empty_like(refb_tensor)
        hdl_b.copy_into_tensor(b_tensor_copy2)
        assert torch.equal(refb_tensor, b_tensor_copy2)

        ssd_buf.from_disk(384)

        assert hdl_a.is_available()
        assert hdl_b.is_available()
        assert hdl_c.is_available()

        assert torch.equal(refa_tensor, hdl_a.get_tensor())
        assert torch.equal(refb_tensor, hdl_b.get_tensor())
        assert torch.equal(refc_tensor, hdl_c.get_tensor())


def test_ssd_buffer_too_small_from_disk():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_tensor = torch.rand((128), dtype=torch.float32)
        ssd_buf = so.SsdBuffer(128, f.name)
        hdl_a = ssd_buf.insert(refa_tensor)
        ssd_buf.to_disk()

        with pytest.raises(RuntimeError):
            ssd_buf.from_disk(127)


def test_ssd_buffer_null_buffer():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_tensor = torch.rand((128), dtype=torch.float32)
        ssd_buf = so.SsdBuffer(128, f.name)
        hdl_a = ssd_buf.insert(refa_tensor)
        ssd_buf.to_disk()

        with pytest.raises(AssertionError):
            ssd_buf.to_disk()

        with pytest.raises(AssertionError):
            hdl_a = ssd_buf.insert(refa_tensor)

        with pytest.raises(AssertionError):
            ssd_buf.can_alloc(128)

        with pytest.raises(AssertionError):
            hdl = ssd_buf.allocate(128)
