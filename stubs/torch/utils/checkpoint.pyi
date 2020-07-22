# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Tuple
from .. import Tensor

def detach_variable(inputs: Tuple[Tensor,...]) -> Tuple[Tensor,...]: ...
