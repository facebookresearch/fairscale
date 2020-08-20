# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from ...optim import Optimizer
from ... import device, Tensor
from typing import Dict

class GradScaler(object):
    def _unscale_grads_(self, optimizer: Optimizer, inv_scale: Tensor, found_inf: Tensor, allow_fp16: bool) -> Dict[device, Tensor]:...