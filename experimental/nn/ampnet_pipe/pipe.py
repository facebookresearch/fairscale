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

"""The AMPnetPipe interface."""

from typing import Any

from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from fairscale.nn.pipe import Pipe
from fairscale.nn.pipe.types import PipelineStyle

from .ampnet import AsyncAMPnetEventLoop

__all__ = ["AMPnetPipe"]


MOVING_DENIED = TypeError("denied to move parameters and buffers, because Pipe should manage device placement")


class AMPnetPipe(Pipe):
    """
        AMPnetPipe is the asynchronous version of the Pipe implementation
        which avoids the bubble issue, by using stale weights and gradients.
        The implementation closely follows the paper: https://arxiv.org/abs/1705.09786
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def interleave(
        self,
        lm_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        transform_logger_object: Any,
        min_update_interval: int = 1,
        weight_prediction: bool = False,
    ) -> None:

        partitions = self.mp_partitions
        n = len(partitions)

        # AMPnet implementation doesn't handle skip_trackers!

        assert self.pipeline.style is PipelineStyle.AsyncSchedule  # type: ignore
        assert self.group
        rank = self.group.rank()

        transport = self.pipeline.transport  # type: ignore
        checkpoint_stop = self.pipeline.checkpoint_stop  # type: ignore
        ampnet_event_loop = AsyncAMPnetEventLoop(
            partitions,
            self.group,
            transport,
            min_update_interval,
            weight_prediction,
            checkpoint_stop,
            self.input_device,
        )

        if rank == 0:
            ampnet_event_loop.event_loop_head_across_minibatches(
                lm_dataloader, criterion, optimizer, transform_logger_object
            )
        elif self.final_stage:
            ampnet_event_loop.event_loop_tail_across_minibatches(
                lm_dataloader, criterion, optimizer, transform_logger_object
            )
        else:
            ampnet_event_loop.event_loop_across_minibatches(
                lm_dataloader, criterion, optimizer, transform_logger_object
            )
