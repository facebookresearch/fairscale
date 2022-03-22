# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum, auto
from functools import reduce
import io
import os
import pickle
from types import TracebackType
from typing import IO, Any, BinaryIO, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL as DEFAULT_PROTOCOL

try:
    from torch.utils._pytree import tree_map
except ImportError:
    # The PyTorch version(<1.9) we test with does not support the tree_map API.
    pass


DEFAULT_CHUNK_SIZE = 2048 * 2048


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

    Class Variables:
        override_directory_path: This variable is used by CheckpointPathContextManager to modify the path to any
            SsdTensorHandles that are saved to a checkpoint via pickling (e.g. torch.save)

    Args:
        shape torch.Size: Shape of the tensor that is represented by the handle.
        dtype: torch.dtype: Dtype of the tensor that is represented by the handle.
        requires_grad: bool: Property of the tensor that is represeneted by the handle.

    Returns:
        A SSDTensorHandle object representing a Tensor.
    """

    override_directory_path: Optional[str] = None

    @staticmethod
    def __new__(
        cls: Type[SsdTensorHandle], shape: torch.Size, dtype: torch.dtype, requires_grad: bool = False
    ) -> SsdTensorHandle:
        r = super(SsdTensorHandle, cls)._make_wrapper_subclass(cls, shape, dtype=dtype, requires_grad=requires_grad)  # type: ignore
        return r

    def __init__(self, shape: torch.Size, dtype: torch.dtype, requires_grad: bool) -> None:
        self._unpickle_f: Optional[Union[BinaryIO, IO[bytes]]] = None

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
        cls, shape: torch.Size, dtype: torch.dtype, filename: str, offset: int = 0, requires_grad: bool = False
    ) -> SsdTensorHandle:
        """Returns a new SsdTensorHandle from a file."""
        handle = cls(shape=shape, dtype=dtype, requires_grad=requires_grad)
        handle.point_to_file(filename, offset=offset)
        return handle

    @classmethod
    def from_tensor(cls: Type[SsdTensorHandle], tensor: torch.Tensor) -> SsdTensorHandle:
        """Returns a new SsdTensorHandle from a tensor."""
        handle = cls(shape=tensor.shape, dtype=tensor.dtype, requires_grad=tensor.requires_grad)
        handle.point_to_tensor(tensor)
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
        self.storage_state = StorageState.ON_DISK

    def point_to_tensor(self, tensor: torch.Tensor) -> None:
        assert self.tensor is None
        assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        self.tensor = tensor
        self.storage_state = StorageState.ON_CPU

    # if resizing a handle that is part of an ssd buffer, care must be taken that the new size
    # doesn't conflict with adjacent handles!
    def point_to_resized_tensor(self, tensor: torch.Tensor) -> None:
        assert self._dtype == tensor.dtype
        self._shape = tensor.shape
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

    def to_file(self, permit_when_tensor_none: bool = False, release_tensor_after_write: bool = True) -> None:
        """Saves the tensor to disk and releases memory if specified."""
        assert self.tensor is not None or permit_when_tensor_none

        if self.tensor is not None:
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

    __torch_function__ = torch._C._disabled_torch_function_impl

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

    @classmethod
    def __unpickle__(
        cls: Type[SsdTensorHandle], shape: torch.Size, dtype: torch.dtype, requires_grad: bool, filename: str
    ) -> SsdTensorHandle:
        result = cls(shape, dtype, requires_grad)
        result.point_to_file(filename, 0)
        result._unpickle_f = io.open(result.filename, "wb")
        return result

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any, Any]:
        byte_iter = None
        filename = self.filename
        if self.override_directory_path is not None:
            head, tail = os.path.split(self.filename)
            filename = os.path.join(self.override_directory_path, tail)
        if self.is_available():
            byte_iter = iter(TensorChunkingIterator(self.tensor))
        else:
            byte_iter = iter(
                FileChunkingIterator(self.filename, expected_size_bytes=self.numel() * self.element_size())
            )
        return (
            self.__unpickle__,  # Callable
            # Args to the callable above
            (self._shape, self._dtype, self.requires_grad, filename),
            None,
            byte_iter,
        )

    def append(self, item: bytes) -> None:
        assert self._unpickle_f
        self._unpickle_f.write(item)

    def extend(self, items: List[bytes]) -> None:
        for i in items:
            self.append(i)


class CheckpointPathContextManager:
    """
    This Context allows the user to override the directory path when pickling an SsdTensorHandle Object.
    It is needed because the filename which the SsdTensorHandle points to (and is used when unpickling)
    is already baked into the pickled data.

    Consider the following example code
        ssd_handle = SsdTensorHandle.from_tensor(ref_tensor)
        ssd_handle.set_file_params('/home/user/handle.bin', 0)
        torch.save(ssd_handle, '/home/user/checkpoint.pkl')
        ssd_handle += 1
        ssd_handle.to_file()
        ssd_handle2 = torch.load('/home/user/checkpoint.pkl')

        print(f"handles are equal: {torch.equals(ssd_handle, ssd_handle2)}")

    One would expect this to print False, however unintuitively it will print True.
    ssd_handle.filename and ssd_handle2.filename are equal. This means that
    when we execute torch.load, we read from the .pkl file and write the result into
    /home/user/handle.bin, clobbering the updated result from `ssd_handle += 1`

    We want to give the user the possibility of not clobbering the data using this
    Context Manager.

        ssd_handle = SsdTensorHandle.from_tensor(ref_tensor)
        ssd_handle.set_file_params('/home/user/handle.bin', 0)
        with CheckpointPathContextManager(override_path='/home/user/checkpoint_data/'):
            torch.save(ssd_handle, '/home/user/checkpoint.pkl')
        ssd_handle += 1
        ssd_handle.to_file()
        ssd_handle2 = torch.load('/home/user/checkpoint.pkl')

        print(f"handles are equal: {torch.equals(ssd_handle, ssd_handle2)}")

    This code results with ssd_handle.filename = '/home/user/handle.bin' and ssd_handle2.filename =
    `/home/user/checkpoint_data/handle.bin'. Therefore the torch.load won't clobber ssd_handle, and
    the printed result is False.

    """

    def __init__(self, override_path: str) -> None:
        self.old_path = SsdTensorHandle.override_directory_path
        self.override_path = override_path

    def __enter__(self) -> None:
        SsdTensorHandle.override_directory_path = self.override_path

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exec_traceback: Optional[TracebackType],
    ) -> None:
        SsdTensorHandle.override_directory_path = self.old_path


# Classes supporting torch.save/load
class TorchSaver:
    def __init__(self) -> None:
        self.pickle_module = DisableMemoizationPicklerModule

    def save(
        self, obj: Any, f: Union[str, os.PathLike, BinaryIO, IO[bytes]], pickle_protocol: int = DEFAULT_PROTOCOL
    ) -> None:
        torch.serialization.save(
            obj, f, self.pickle_module, pickle_protocol=pickle_protocol, _use_new_zipfile_serialization=False
        )


class SsdParameter(SsdTensorHandle, torch.nn.Parameter):
    @classmethod
    def from_tensor(cls: Type[SsdParameter], tensor: SsdTensorHandle) -> SsdParameter:  # type: ignore
        r = cls(tensor.shape, tensor.dtype, tensor.requires_grad)
        r.point_to_tensor(tensor)
        return r

    @staticmethod
    def __new__(
        cls: Type[SsdParameter], shape: torch.Size, dtype: torch.dtype, requires_grad: bool = True
    ) -> SsdParameter:
        r = super(SsdParameter, cls).__new__(cls, shape, dtype=dtype, requires_grad=requires_grad)
        return r  # type: ignore

    def __init__(self, shape: torch.Size, dtype: torch.dtype, requires_grad: bool = True) -> None:
        super(SsdParameter, self).__init__(shape, dtype, requires_grad)


class SsdFlatParameter(SsdParameter):
    """A parameter that is initialized from a list of parameters and can be
    turned into a list of views as needed.

    This class should eventually be moved to fairscale/nn/misc/flatten_params_wrapper.py
    """

    def __new__(
        cls: Type[SsdFlatParameter], shapes: Sequence[torch.Size], dtype: torch.dtype, requires_grad: bool = True
    ) -> SsdFlatParameter:
        """Make an object using the parent's __new__ function."""

        # A empty of non-list input doesn't make sense.
        if not isinstance(shapes, (list, tuple)) or len(shapes) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        size = sum([np.prod(s) for s in shapes])
        r = super(SsdFlatParameter, cls).__new__(cls, torch.Size((size,)), dtype=dtype, requires_grad=requires_grad)
        return r  # type: ignore

    def __init__(self, shapes: Sequence[torch.Size], dtype: torch.dtype, requires_grad: bool = True):
        """Initialize the _param_numels and _param_shapes lists."""
        self._param_shapes = shapes
        self._param_numels = [np.prod(s) for s in shapes]
        total_numels = sum(self._param_numels)
        assert (
            self.numel() <= total_numels
        ), f"Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}"

        # These are set by FPW class below, not by this class itself.
        self._param_infos: List[Tuple[str, torch.nn.Module, str]] = []
        self._shared_param_infos: List[Tuple[str, str, torch.nn.Module, str, torch.nn.Module, str]] = []

        super(SsdFlatParameter, self).__init__(
            shape=torch.Size((total_numels,)), dtype=dtype, requires_grad=requires_grad
        )

    def get_param_views(self, external_data: Optional[torch.Tensor] = None) -> Iterator[torch.Tensor]:
        """Return a generator of views that map to the original parameters."""
        # Note, self.data could be sharded, so its numel is <= to the sum.
        """
        assert self.data.numel() <= sum(
            self._param_numels
        ), f"Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}"
        """
        if external_data is not None:
            if external_data.numel() != sum(self._param_numels):
                raise ValueError(
                    f"Incorrect numel of supplied data: got {external_data.numel()} but expected {sum(self._param_numels)}"
                )
            return (t.view(s) for (t, s) in zip(external_data.split(self._param_numels), self._param_shapes))
        else:
            return (t.view(s) for (t, s) in zip(self.split(self._param_numels), self._param_shapes))

    def metadata(self) -> Tuple[List[str], Sequence[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [".".join([m, n]) if m else n for (m, _, n) in self._param_infos]
        return names, self._param_shapes, self._param_numels

    @classmethod
    def from_tensors(
        cls: Type[SsdFlatParameter], tensors: Sequence[torch.Tensor], direct_to_file: bool = False
    ) -> "SsdFlatParameter":
        """Returns a new SsdFlatParameter from a sequence of tensors."""
        assert (
            len(tensors) > 0
        ), "SsdFlatParameter.from_tensors must be called with at least one tensor in the tensors argument"

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and (2) making a module
        # heirarchy flat (using a single tensor to replace a tree of tensors). Therefore,
        # adding back nesting and heirarchy is counter-productive. If nesting is encountered
        # in the future, the reasonable thing to do is likely for the top level SsdFlatParameter to
        # absorb the nested one and keep the result flat, free from hierarchy.
        if any(isinstance(t, SsdFlatParameter) for t in tensors):
            raise ValueError("Nesting SsdFlatParameter is not supported")

        handle = cls(shapes=[t.size() for t in tensors], dtype=tensors[0].dtype, requires_grad=tensors[0].requires_grad)
        if direct_to_file:
            assert False, "direct_to_file not implemented yet"
            pass
        else:
            tensor = torch.cat(
                [t.detach().reshape(-1) if isinstance(t, torch.nn.Parameter) else t.reshape(-1) for t in tensors], 0
            )
            handle.point_to_tensor(tensor)
        return handle

    @classmethod
    def __unpickle_SFP__(
        cls: Type[SsdFlatParameter],
        shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        requires_grad: bool,
        filename: str,
    ) -> SsdFlatParameter:
        result = cls(shapes, dtype, requires_grad)
        result.point_to_file(filename, 0)
        result._unpickle_f = io.open(result.filename, "wb")
        return result

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any, Any]:
        byte_iter = None
        filename = self.filename
        if self.override_directory_path is not None:
            head, tail = os.path.split(self.filename)
            filename = os.path.join(self.override_directory_path, tail)
        if self.is_available():
            byte_iter = iter(TensorChunkingIterator(self.tensor))
        else:
            byte_iter = iter(
                FileChunkingIterator(self.filename, expected_size_bytes=self.numel() * self.element_size())
            )
        return (
            self.__unpickle_SFP__,  # Callable
            # Args to the callable above
            (self._param_shapes, self._dtype, self.requires_grad, filename),
            None,
            byte_iter,
        )


class DisableMemoizationPicklerModule:
    @classmethod
    def Pickler(cls, data_buf: io.BytesIO, protocol: int) -> pickle.Pickler:
        p = pickle.Pickler(data_buf, protocol)
        p.fast = True
        return p

    @classmethod
    def dump(cls, obj: Any, f: io.BytesIO, protocol: int) -> None:
        pickle.dump(obj, f, protocol)


class TensorChunkingIterator:
    """
    chunk_size_bytes determines how large each chunk that we break the tensor
    into. It is important to consider limiting the size because by when
    python unpickles an object, by default it will read up to 1000 list
    elements at a time. So memory usage while unpickling will be on the
    order of O(min(file_size, 1000 * chunk_size_bytes)).
    """

    def __init__(self, tensor: torch.Tensor, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> None:

        self.tensor = tensor
        self.chunk_size_bytes = chunk_size_bytes

    def __iter__(self) -> Iterator[bytes]:

        self.num_chunks = _get_num_chunks(self.tensor, self.chunk_size_bytes)
        self.num_chunks_read = 0
        return self

    def __next__(self) -> bytes:
        if self.num_chunks_read >= self.num_chunks:
            raise StopIteration
        next_chunk = _tensor_to_bytes_chunks(
            self.tensor, chunk_idx=self.num_chunks_read, chunk_size_bytes=self.chunk_size_bytes
        )

        self.num_chunks_read += 1

        return next_chunk


class FileChunkingIterator:
    """
    chunk_size_bytes determines how large each chunk that we break the file
    into. It is important to consider limiting the size because by when
    python unpickles an object, by default it will read up to 1000 list
    elements at a time. So memory usage while unpickling will be on the
    order of O(min(file_size, 1000 * chunk_size_bytes)).
    """

    def __init__(
        self, filename: str, expected_size_bytes: int = -1, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE
    ) -> None:
        self.filename = filename
        self.file: Optional[Union[BinaryIO, IO[bytes]]] = None
        self.chunk_size_bytes = chunk_size_bytes
        self.expected_size_bytes = expected_size_bytes

    def __iter__(self) -> Iterator[bytes]:

        if self.expected_size_bytes != -1:
            file_size = os.stat(self.filename).st_size
            assert (
                file_size == self.expected_size_bytes
            ), f"FileChunkingIterator Failed, expecting file to be of size: {self.expected_size_bytes} but got {file_size}"
        self.file = io.open(self.filename, "rb", buffering=0)
        self.num_chunks_read = 0
        return self

    def __next__(self) -> bytes:
        assert self.file
        next_chunk = self.file.read(self.chunk_size_bytes)

        if len(next_chunk) == 0:
            raise StopIteration
        self.num_chunks_read += 1

        return next_chunk


torch_saver = TorchSaver()
