# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Generator

from .grad_scaler import GradScaler as GradScaler

class autocast:
    def __init__(self, enabled=True) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
