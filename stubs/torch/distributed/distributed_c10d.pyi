# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, List, Union, Optional

from . import ProcessGroup

def _get_global_rank(group: ProcessGroup, rank: int) -> int: ...

def _get_default_group() -> ProcessGroup: ...