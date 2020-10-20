# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Tuple
from .. import Tensor
from torch.nn.modules.module import Module

def detach_variable(inputs: Tuple[Tensor,...]) -> Tuple[Tensor,...]: ...
def checkpoint(function: Module, *args, **kwargs): ...
