# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, List

class Future:
    def wait(self) -> Any: ...

def wait_all(futures: List[Future]) -> List[Any]: ...
