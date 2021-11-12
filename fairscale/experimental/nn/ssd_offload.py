# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum, auto
from functools import reduce
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from torch.utils._pytree import tree_map
except ImportError:
    # The PyTorch version(<1.9) we test with does not support the tree_map API.
    pass


DEFAULT_CHUNK_SIZE = 1024 * 1024


def _get_num_chunks(input_tensor: torch.Tensor, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> int:
    """Returns the number of chunks that the given tensor can be divided into."""
    size_in_bytes = input_tensor.nelement() * input_tensor.element_size()
    num_chunks = (size_in_bytes + (chunk_size_bytes - 1)) // chunk_size_bytes
    return num_chunks


def _tensor_to_bytes_chunks(
    input_tensor: torch.Tensor, chunk_idx: int, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE
) -> bytes:
    """Converts the given tensor into a chunked array containing chunk_size_bytes."""
    size_in_bytes = input_tensor.nelement() * input_tensor.element_size()
    assert chunk_idx < _get_num_chunks(input_tensor, chunk_size_bytes)
    input_tensor_np = input_tensor.detach().numpy().view(np.uint8).reshape(-1)
    chunk_start = chunk_idx * chunk_size_bytes
    chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
    return input_tensor_np[chunk_start:chunk_end].tobytes()


def write(input_tensor: torch.Tensor, filename: str, file_offset_bytes: int = 0) -> None:
    """Populates the file with the data stored in the given tensor."""
    num_chunks = _get_num_chunks(input_tensor)
    file_flags = "r+b" if os.path.exists(filename) else "wb"
    with open(filename, file_flags) as f:
        f.seek(file_offset_bytes)
        for i in range(num_chunks):
            f.write(_tensor_to_bytes_chunks(input_tensor, i))


def read(input_tensor: torch.Tensor, filename: str, file_offset_bytes: int = 0) -> None:
    """Populates the given tensor with the data stored in a file."""
    size_in_bytes = input_tensor.nelement() * input_tensor.element_size()
    chunk_size_bytes = DEFAULT_CHUNK_SIZE
    num_chunks = _get_num_chunks(input_tensor)
    input_tensor_np = input_tensor.detach().numpy()
    input_tensor_mv = memoryview(input_tensor_np.view(dtype=np.uint8).reshape(-1))
    with io.open(filename, "rb") as f:
        f.seek(file_offset_bytes)
        for i in range(num_chunks):
            chunk_start = i * chunk_size_bytes
            chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
            data_read = f.readinto(input_tensor_mv[chunk_start:chunk_end])
            assert data_read == chunk_end - chunk_start


class StorageState(Enum):
    """
    Simple enum to indicate whether the tensor handle is pointing
    to data on disk or memory. This is useful for asserting on
    whether the tensor is available for operations or if it needs
    to be moved from disk to CPU or device.
    """

    UNALLOCATED = auto()
    ON_DISK = auto()
    ON_CPU = auto()


class SsdTensorHandle(torch.Tensor):
    """
    This class extends from torch.Tensor and represents a Tensor which is backed by SSD storage.
    The SsdTensorHandle object can point to a file or a tensor and there are corresponding functions to read
    data into the tensor that is an attribute of the SsdTensorHandle object or write the tensor to file. At any
    point in time the Tensor may be in memory or on disk.

    Args:
        shape Tuple[int, ...]: Shape of the tensor that is represented by the handle.
        dtype: torch.dtype: Dtype of the tensor that is represented by the handle.
        requires_grad: bool: Property of the tensor that is represeneted by the handle.

    Returns:
        A SSDTensorHandle object representing a Tensor.
    """

    @staticmethod
    def __new__(
        cls: SsdTensorHandle, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool = False
    ) -> SsdTensorHandle:
        r = torch.Tensor._make_wrapper_subclass(cls, shape, dtype=dtype, requires_grad=requires_grad)  # type: ignore
        return r

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool) -> None:
        self._shape = shape
        if len(shape) == 0:
            self._numel = 0
        else:
            self._numel = reduce((lambda x, y: x * y), shape)
        self._dtype = dtype
        # valid if offloaded to file
        self.filename = ""
        self.offset = -1
        # valid if loaded to memory
        self.tensor: Optional[torch.Tensor] = None
        self.requires_grad = requires_grad
        self.storage_state = StorageState.UNALLOCATED

    @classmethod
    def from_file(
        cls, shape: Tuple[int, ...], dtype: torch.dtype, filename: str, requires_grad: bool = False
    ) -> SsdTensorHandle:
        """Returns a new SsdTensorHandle from a file."""
        handle = cls(shape=shape, dtype=dtype, requires_grad=requires_grad)
        handle.filename = filename
        handle.storage_state = StorageState.ON_DISK
        return handle

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> SsdTensorHandle:
        """Returns a new SsdTensorHandle from a tensor."""
        handle = cls(shape=tensor.shape, dtype=tensor.dtype, requires_grad=tensor.requires_grad)
        handle.tensor = tensor
        handle.storage_state = StorageState.ON_CPU
        return handle

    def is_available(self) -> bool:
        return self.tensor is not None

    def get_tensor(self) -> torch.Tensor:
        assert self.tensor is not None
        return self.tensor

    def set_file_params(self, filename: str, offset: int) -> None:
        self.filename = filename
        self.offset = offset

    def point_to_file(self, filename: str, offset: int) -> None:
        self.set_file_params(filename, offset)
        self.tensor = None

    def point_to_tensor(self, tensor: torch.Tensor) -> None:
        assert self.tensor is None
        assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Returns the tensor represented by the SsdTensorHandle object.

        If the tensor is on disk, it is copied into the tensor attribute and returned.
        """
        if self.tensor is not None:
            return self.tensor
        else:
            result_tensor = torch.empty(size=self._shape, dtype=self._dtype, requires_grad=self.requires_grad)
            self.copy_into_tensor(result_tensor)
            self.tensor = result_tensor
            self.storage_state = StorageState.ON_CPU
            return self.tensor

    def to_file(self, release_tensor_after_write: bool = True) -> None:
        """Saves the tensor to disk and releases memory if specified."""
        assert self.tensor is not None
        write(self.tensor, self.filename, self.offset * self.tensor.element_size())
        if release_tensor_after_write:
            self.tensor = None
            self.storage_state = StorageState.ON_DISK

    def copy_into_tensor(self, tensor: torch.Tensor) -> None:
        """Copies SsdTensorHandle's data into the given tensor.

        If the tensor is in memory, this function copies the data
        into the passed in tensor. Otherwise, it reads from file into tensor,
        using the read() function.
        This does not modify modify self.tensor unlike the to_tensor()
        function. This can be useful for calls like named_parameters() when
        the tensor is already offloaded to disk.
        """
        assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        if self.tensor is not None:
            tensor.copy_(self.tensor)
        else:
            read(tensor, self.filename, self.offset * tensor.element_size())

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        """Intercepts all operations performed on this handle object.

        Before any operation, the tensor attribute is unwrapped from the handle
        and used in the operation. We maintain a refernce to the tensor and its current
        versions to track if modifications have been made. If we detect changes to the
        tensor, we write it to the file maintained by the Handle.
        """
        ssd_tensor_handles = []

        def unwrap(e: Any) -> torch.Tensor:
            if isinstance(e, SsdTensorHandle):
                t = e.to_tensor()
                ssd_tensor_handles.append((e, t._version))  # type: ignore
                return t
            else:
                return e

        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for e, saved_version in ssd_tensor_handles:
            inplace_is_this_tensor = func.__name__[-1] == "_" and e is args[0]
            out_is_this_tensor = False if "out" not in kwargs else e is kwargs["out"]
            if inplace_is_this_tensor or out_is_this_tensor:
                e.to_file()
        return r


class SsdBuffer:
    """
    The SsdBuffer represents a single buffer containing a list of tensors. Each of the
    tensors are represented by a `SsdTensorHandle`.

    Args:
        num_elems (int): Dictates the size of the 1-D tensor.
        dtype (torch.dtype): Dtype of the buffer.
    """

    def __init__(self, num_elems: int, filename: str, dtype: torch.dtype = torch.float32) -> None:
        self.buffer: torch.Tensor = torch.empty((num_elems,), dtype=dtype)
        self.filename = filename
        self.offset = 0
        self.tensors: Dict[int, SsdTensorHandle] = {}
        self.storage_state = StorageState.ON_CPU

    def allocate(self, num_elems: int) -> SsdTensorHandle:
        """Allocates a new tensor handle of size num_elems."""
        assert num_elems > 0
        assert self.storage_state == StorageState.ON_CPU, self.storage_state
        assert self.can_alloc(num_elems)

        tensor = self.buffer.narrow(0, self.offset, num_elems)

        tensor_offset = self.offset
        handle = SsdTensorHandle.from_tensor(tensor)
        self.tensors[tensor_offset] = handle
        handle.set_file_params(self.filename, tensor_offset)
        self.offset += num_elems

        return handle

    def insert(self, tensor: torch.Tensor) -> SsdTensorHandle:
        """Insert a new tensor by allocating memory and creating a corresponding handle."""
        assert self.storage_state == StorageState.ON_CPU, self.storage_state
        # For the non sharded case, the tensor will not be flattened
        tensor = tensor.reshape(-1)
        assert self.buffer.dtype == tensor.dtype
        handle = self.allocate(tensor.numel())
        handle.get_tensor().copy_(tensor)
        return handle

    def can_alloc(self, num_elems: int) -> bool:
        """Verify that you can allocate a tensor within the bounds
        of the larger SsdBuffer memory buffer."""
        assert self.storage_state == StorageState.ON_CPU, self.storage_state
        return (self.offset + num_elems) <= self.buffer.numel()

    def get_tensors(self) -> List[SsdTensorHandle]:
        """Returns the list of tensor handles in SsdBuffer."""
        return [t for t in self.tensors.values()]

    def to_disk(self) -> None:
        """Writes all tensors backed by handles to disk."""
        if self.storage_state == StorageState.ON_DISK:
            return
        assert self.storage_state == StorageState.ON_CPU, self.storage_state
        # We use `narrow` so that we write valid tensors that have been allocated
        # as opposed to the entire SSD buffer.
        valid_data = self.buffer.narrow(0, 0, self.offset)
        write(valid_data, self.filename)

        # Remove all Tensor references
        for offset, t in self.tensors.items():
            t.point_to_file(self.filename, offset)

        # TODO(anj-s): Setting this to None does not result in GC picking
        # this reference up.
        self.buffer = torch.empty((1))
        self.storage_state = StorageState.ON_DISK

    def from_disk(self, num_elems: int, dtype: torch.dtype = torch.float32) -> None:
        """Reads all tensors backed by handles into memory."""
        if self.storage_state == StorageState.ON_CPU:
            return
        assert self.storage_state == StorageState.ON_DISK, self.storage_state
        if num_elems < self.offset:
            raise RuntimeError(
                f"Attempted to load from file ssdbuffer of size: {self.offset} into a buffer that is of size: {num_elems}"
            )
        self.buffer = torch.empty((num_elems,), dtype=dtype)
        valid_data = self.buffer.narrow(0, 0, self.offset)
        read(valid_data, self.filename)

        for offset, t in self.tensors.items():
            t.point_to_tensor(self.buffer.narrow(0, t.offset, t._numel))

        self.storage_state = StorageState.ON_CPU
