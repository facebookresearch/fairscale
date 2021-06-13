# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

__all__: List[str] = []


def torch_version() -> List[int]:
    return [int(x) for x in torch.__version__.split(".")[:2]]
