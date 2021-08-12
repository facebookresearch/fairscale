# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Pipe implementation in PyTorch."""
from .async_pipe import AsyncPipe
from .checkpoint import is_checkpointing, is_recomputing
from .pipe import Pipe
from .rpc import PipeRPCWrapper
from .types import LazyModule

__all__ = ["Pipe", "is_checkpointing", "is_recomputing", "LazyModule"]
