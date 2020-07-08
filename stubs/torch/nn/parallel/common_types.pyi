# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Union, Sequence
from ... import device

_device_t = Union[int, device]
_devices_t = Sequence[_device_t]
