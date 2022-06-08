# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
import hashlib
import os
import pathlib
import sys
from typing import Union


def create_dir(dir_path: Union[str, pathlib.Path], exception_msg: str = "") -> None:
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        sys.stderr.write(exception_msg + "\n")
        sys.exit(ExitCode.FILE_EXISTS_ERROR)


def create_file(
    file_path: Union[str, pathlib.Path], exception_msg: str = "An exception occured while creating "
) -> None:
    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            pass
    else:
        sys.stderr.write(exception_msg + f"{file_path}: {repr(FileExistsError)}\n")
        sys.exit(ExitCode.FILE_EXISTS_ERROR)


def write_to_file(file_path: Union[str, pathlib.Path], msg: str) -> None:
    with open(file_path, "a") as file:
        file.write(msg)


def get_sha1_hash(file_path: Union[str, pathlib.Path]) -> str:
    BUF_SIZE = 104857600  # Reading file in 100MB chunks

    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


class ExitCode(Enum):
    CLEAN = 0
    FILE_EXISTS_ERROR = 1
    FILE_DOES_NOT_EXIST_ERROR = 2

    ERROR = -1  # unknown errors
