# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Callable, Optional, Tuple

from torch import Tensor

def spawn(
    fn: Callable[..., Any],
    args: Tuple[Optional[Any], ...] = (),
    nprocs: int = 1,
    join: bool = True,
    daemon: bool = False,
    start_method: str = "spawn",
): ...

