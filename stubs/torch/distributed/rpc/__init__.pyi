# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Union, Callable, Optional
from torch.futures import Future


class RRef:
    ...


class WorkerInfo:
    ...


def rpc_async(
    to: Union[str, WorkerInfo],
    func: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    timeout=-1.0,
) -> Future:
    ...


def rpc_sync(
    to: Union[str, WorkerInfo],
    func: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    timeout=-1.0,
) -> None:
    ...
