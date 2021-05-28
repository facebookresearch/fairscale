# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import abc
import io
from math import inf
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist


class Workhandle:
    def __init__(self, handle: Any, callback: Optional[Callable]) -> None:
        self.handle = handle
        self.callback = callback


def get_global_rank(group: Any, rank: int) -> int:
    if group is dist.group.WORLD:
        return rank

    return dist.distributed_c10d._get_global_rank(group, rank)


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

    if isinstance(value, abc.Mapping):
        device_val: Dict[str, Any] = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(val, non_blocking=non_blocking, device=device)

        return device_val

    return value


# backward compatibility - this is needed for torch 1.5 which does not expose this functionality
# FIXME: to be dropped alongside torch1.5 support, when time comes
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


def calc_grad_norm(parameters: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    r"""Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda par: par.grad is not None, parameters))

    if len(parameters) == 0:
        return torch.tensor(0.0)
    p = float(p)
    if p == inf:
        local_norm = max(par.grad.detach().abs().max() for par in parameters)  # type: ignore
    else:
        # Compute the norm in full precision no matter what
        local_norm = torch.norm(torch.stack([torch.norm(par.grad.detach(), p, dtype=torch.float32) for par in parameters]), p).to(dtype=parameters[0].dtype)  # type: ignore
    return local_norm
