# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
import hashlib
from pathlib import Path
from typing import Union


def get_sha1_hash(file_path: Union[str, Path]) -> str:
    SHA1_BUF_SIZE = 104857600  # Reading file in 100MB chunks

    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(SHA1_BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


class ExitCode(Enum):
    """Collections of the Exit codes as an Enum class"""

    CLEAN = 0
    FILE_EXISTS_ERROR = 1
    FILE_DOES_NOT_EXIST_ERROR = 2

    ERROR = -1  # unknown errors
