# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Import most common subpackages
#
# NOTE: we don't maintain any public APIs in both experimental and fair_dev
#       sub-modules. Code in them are experimental or for developer only. They
#       can be changed, removed, anytime.
################################################################################

from typing import List

from . import nn
from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])
__all__: List[str] = []
