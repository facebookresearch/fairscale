# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._six import container_abcs
import torch.distributed as dist


class Workhandle:
    def __init__(self, handle: Any, callback: Optional[Callable]) -> None:
        self.handle = handle
        self.callback = callback


class FlatParam:
    def __init__(self, tensor: torch.Tensor, start: int, stop: int) -> None:
        self.param = tensor
        self.start = start
        self.stop = stop


class Bucket:
    """
    Helper class to simplify the handling of broadcast or reduce buckets
    """

    def __init__(self, buffer: torch.Tensor) -> None:
        # The actual flat tensor
        self.buffer = buffer
        self.max_size = buffer.numel()

        # Handles to the params and their position in this tensor, can be useful for a callback
        self.params: List[FlatParam] = []

        # Optional callback, possibly to unwrap the bucket
        self.callback: Optional[Callable] = None

        # Current status for this buffer
        self.current_offset = 0
        self.max_offset = 0
        self.global_ref_rank = -1  # Either the destination or the src rank, if reducing or broadcasting for instance
        self.global_rank = -1
        self.gradients_based = False

    def unroll(self) -> None:
        """
        Dsitribute the contents of the flat buffer back to the attached parameters
        """

        for flat in self.params:
            if self.global_ref_rank != self.global_rank and self.gradients_based:
                # this rank is not the owner, release the grad
                flat.param.grad = None
            else:
                if self.gradients_based:
                    # this rank is the owner, unroll the results
                    assert flat.param.grad is not None

                    flat.param.grad.data.copy_(
                        self.buffer[flat.start : flat.stop].view_as(flat.param.data), non_blocking=True
                    )
                else:
                    flat.param.data.copy_(
                        self.buffer[flat.start : flat.stop].view_as(flat.param.data), non_blocking=True
                    )

        self.reset()

    def reset(self) -> None:
        """ empty the bucket """
        self.current_offset = 0
        self.params.clear()

    def append(self, tensor: torch.Tensor, use_gradient: bool = False) -> bool:
        """ add a tensor to the bucket """

        end = self.current_offset + tensor.numel()
        self.gradients_based = use_gradient

        if end > self.max_size:
            return False

        if use_gradient:
            assert tensor.grad is not None

        data_source = tensor.grad.data if use_gradient else tensor.data  # type: ignore    # mypy is drunk
        self.buffer[self.current_offset : end].copy_(data_source.view(-1))
        self.params.append(FlatParam(tensor=tensor, start=self.current_offset, stop=end))
        self.current_offset = end
        return True

    def full(self) -> bool:
        """ is the bucket full ? """
        return self.current_offset == self.max_offset


# Credits:  classy_vision/generic/distributed_util.py
def recursive_copy_to_device(value: Any, non_blocking: bool, device: torch.device) -> Any:
    """
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.

    NOTE:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = []
        for val in value:
            values.append(recursive_copy_to_device(val, non_blocking=non_blocking, device=device))

        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, container_abcs.Mapping):
        device_val: Dict[str, Any] = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(val, non_blocking=non_blocking, device=device)

        return device_val

    return value


def broadcast_object(
    obj: Any, src_rank: int, group: object = dist.group.WORLD, dist_device: torch.device = torch.device("cpu")
) -> Any:
    """
    Either broadcast from master to the fleet (default),
    or use the src setting as the original rank.
    """

    if dist.get_rank() == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(dist_device)
        data_send_tensor = torch.ByteTensor(data).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0]).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=dist_device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=dist_device)
    return obj
