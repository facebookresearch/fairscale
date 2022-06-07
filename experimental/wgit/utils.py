# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
import hashlib
import os
import sys


def create_dir(dir_path, exception_msg=""):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        sys.stderr.write(exception_msg + "\n")
        sys.exit(ExitCode.FILE_EXISTS_ERROR)


def create_file(file_path, exception_msg="An exception occured while creating "):
    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            pass
    else:
        sys.stderr.write(exception_msg + f"{file_path}: {repr(FileExistsError)}\n")
        sys.exit(ExitCode.FILE_EXISTS_ERROR)


def write_to_file(file_path, msg):
    with open(file_path, "a") as file:
        file.write(msg)


def get_sha1_hash(file_path):
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
