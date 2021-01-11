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

"""The Pipe interface."""
from collections import OrderedDict
import itertools
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, cast
import warnings

from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from fairscale.nn.pipe import Pipe 
from .ampnet import AsyncAMPnetEventLoop

from fairscale.nn.pipe.types import PipelineStyle

__all__ = ["AMPnetPipe"]


MOVING_DENIED = TypeError("denied to move parameters and buffers, because Pipe should manage device placement")


class AMPnetPipe(Pipe):
    def __init__(self, **kwargs)-> None:
        super().__init__(**kwargs)

    def interleave(
        self,
        lm_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        vocab_size: int,
        weight_prediction: bool = False,
    ) -> None:
#        self.pipeline.run_ampnet(lm_dataloader, criterion, optimizer, vocab_size, weight_prediction)  # type: ignore

        partitions = self.mp_partitions
        n = len(partitions)

        # AMPnet implementation doesn't handle skip_trackers!

        assert self.style is PipelineStyle.AsyncSchedule
        assert self.group
        rank = self.group.rank()

        min_update_interval = 10

        ampnet_event_loop = AsyncAMPnetEventLoop(
            partitions,
            self.group,
            self.transport,
            min_update_interval,
            weight_prediction,
            self.checkpoint_stop,
            self.input_device,
        )

        if rank == 0:
            ampnet_event_loop.event_loop_head_across_minibatches(lm_dataloader, criterion, optimizer, vocab_size)
        elif self.final_stage:
            ampnet_event_loop.event_loop_tail_across_minibatches(lm_dataloader, criterion, optimizer, vocab_size)
        else:
            ampnet_event_loop.event_loop_across_minibatches(lm_dataloader, criterion, optimizer, vocab_size)

