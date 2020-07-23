# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import Any

import torch
from torch._six import container_abcs
import torch.distributed as dist


def recursive_copy_to_device(
    value: Any, non_blocking: bool, device: torch.device
) -> Any:
    """
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.

    Note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        device_val = []
        for val in value:
            device_val.append(
                recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            )

        return device_val if isinstance(value, list) else tuple(device_val)

    if isinstance(value, container_abcs.Mapping):
        device_val = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )

        return device_val

    return value


def broadcast_object(obj: Any, src_rank: int) -> Any:
    """
    Either broadcast from master to the fleet (default),
    or use the src setting as the original rank.

    The object needs to be on the appropriate device, depending on the
    backend being used.
    """
    if dist.get_rank() == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)])
        data_tensor = torch.ByteTensor(data)
        dist.broadcast(length_tensor, src=src_rank)
        dist.broadcast(data_tensor, src=src_rank)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0])
        dist.broadcast(length_tensor, src=src_rank)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8)
        dist.broadcast(data_tensor, src=src_rank)
        buffer = io.BytesIO(data_tensor.numpy())
        obj = torch.load(buffer)
    return obj
