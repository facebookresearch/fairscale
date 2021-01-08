# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

import torch
from torch._six import container_abcs


class Workhandle:
    def __init__(self, handle: Any, callback: Optional[Callable]) -> None:
        self.handle = handle
        self.callback = callback


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
