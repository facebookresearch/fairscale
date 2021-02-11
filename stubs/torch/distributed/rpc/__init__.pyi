# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Union, Callable, Optional, Any
from torch.futures import Future

class RRef: ...
class WorkerInfo: ...

class BackendType:
    TENSORPIPE: Any
    PROCESS_GROUP: Any

def TensorPipeRpcBackendOptions(init_method: str) -> Any: ...
def ProcessGroupRpcBackendOptions(init_method: str) -> Any: ...
def rpc_async(
    to: Union[str, WorkerInfo],
    func: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    timeout=-1.0,
) -> Future: ...
def rpc_sync(
    to: Union[str, WorkerInfo],
    func: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    timeout=-1.0,
) -> None: ...
def init_rpc(
    name: str,
    backend: Optional[Any] = None,
    rank: int = -1,
    world_size: Optional[int] = None,
    rpc_backend_options: Optional[Any] = None,
) -> None: ...
def shutdown(graceful: Optional[bool] = True) -> None: ...
