# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from .repo import Repo
from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])
__all__: List[str] = []
