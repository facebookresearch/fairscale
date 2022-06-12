# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class ExitCode(Enum):
    """Collections of the Exit codes as an Enum class"""

    CLEAN = 0
    FILE_EXISTS_ERROR = 1
    FILE_DOES_NOT_EXIST_ERROR = 2
