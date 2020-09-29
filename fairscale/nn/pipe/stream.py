# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for eliminating boilerplate code to handle abstract streams with
CPU device.
"""
from contextlib import contextmanager
from typing import Generator, List, Optional, Union, cast

import torch

__all__: List[str] = []


class CPUStreamType:
    pass


# The placeholder on place of streams for the CPU device instead of CUDA.
CPUStream = CPUStreamType()

# It represents both CUDA streams and the CPU stream.
AbstractStream = Union[torch.cuda.Stream, CPUStreamType]


def new_stream(device: torch.device) -> AbstractStream:
    """Creates a new stream for either CPU or CUDA device."""
    if device.type != "cuda":
        return CPUStream
    return torch.cuda.Stream(device)


def current_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != "cuda":
        return CPUStream
    return torch.cuda.current_stream(device)


def default_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != "cuda":
        return CPUStream
    return torch.cuda.default_stream(device)


@contextmanager
def use_device(device: torch.device) -> Generator[None, None, None]:
    """:func:`torch.cuda.device` for either CPU or CUDA device."""
    if device.type != "cuda":
        yield
        return

    with torch.cuda.device(device):
        yield


@contextmanager
def use_stream(stream: Optional[AbstractStream]) -> Generator[None, None, None]:
    """:func:`torch.cuda.stream` for either CPU or CUDA stream."""
    if not stream:
        yield
        return

    if not is_cuda(stream):
        yield
        return

    with torch.cuda.stream(as_cuda(stream)):
        yield


def get_device(stream: AbstractStream) -> torch.device:
    """Gets the device from CPU or CUDA stream."""
    if is_cuda(stream):
        return as_cuda(stream).device
    return torch.device("cpu")


def wait_stream(source: AbstractStream, target: AbstractStream) -> None:
    """:meth:`torch.cuda.Stream.wait_stream` for either CPU or CUDA stream. It
    makes the source stream wait until the target stream completes work queued.
    """
    if is_cuda(target):
        if is_cuda(source):
            # A CUDA stream waits another CUDA stream.
            as_cuda(source).wait_stream(as_cuda(target))
        else:
            # CPU waits a CUDA stream.
            as_cuda(target).synchronize()

    # If the target is CPU, synchronization is not required.


def record_stream(tensor: torch.Tensor, stream: AbstractStream) -> None:
    """:meth:`torch.Tensor.record_stream` for either CPU or CUDA stream."""
    if is_cuda(stream):
        # NOTE(sublee): record_stream() on a shifted view tensor throws
        # RuntimeError in PyTorch 1.1.0, and does nothing in 1.2.0. To safely
        # protect the tensor against unexpected reallocation, here we use a
        # temporal tensor associated with the same storage without shifting as
        # a workaround.
        #
        # Issue: https://github.com/pytorch/pytorch/issues/27366
        #
        tensor = tensor.new_empty([0]).set_(tensor.storage())

        tensor.record_stream(as_cuda(stream))


def is_cuda(stream: Optional[AbstractStream]) -> bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def as_cuda(stream: AbstractStream) -> torch.cuda.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.cuda.Stream, stream)
