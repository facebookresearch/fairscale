from enum import Enum


class ExitCode(Enum):
    CLEAN = 0
    FILE_EXISTS_ERROR = 1
    FILE_DOES_NOT_EXIST_ERROR = 2

    ERROR = -1  # unknown errors
