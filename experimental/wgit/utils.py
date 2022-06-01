from enum import Enum


class ExitCode(Enum):
    CLEAN = 0
    FILE_EXISTS_ERROR = 1

    ERROR = -1  # unknown errors
