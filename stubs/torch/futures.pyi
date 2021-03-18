# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, List

class Future:
    def wait(self) -> Any: ...
    
    def then(self, callback: Any) -> Future : ...

def collect_all(futures : List[Future]) -> Future : ...
