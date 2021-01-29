# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from .multiprocess_pipe import MultiProcessPipe
from .types import PipelineStyle


class AsyncPipe(MultiProcessPipe):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, style=PipelineStyle.MultiProcess, **kwargs)
