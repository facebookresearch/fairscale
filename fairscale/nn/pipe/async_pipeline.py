# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from threading import Event
from typing import Dict, List, Optional, Union

import torch

from .async_schedule import AsyncEventLoop, ModuleWrapper
from .messages import MakeTransport
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals


class AsyncPipeline:
    """The async pipeline parallelism for Pipe."""

    def __init__(
        self,
        partitions: List[ModuleWrapper],
        skip_layout: SkipLayout,
        checkpoint_stop: int,
        group: torch.distributed.ProcessGroup,
        *,
        worker_map: Optional[Dict[int, str]] = None,
        input_device: Union[None, int, str, torch.device] = None,
        final_stage: bool = False,
    ) -> None:
        self.partitions = partitions
        self.skip_layout = skip_layout
        self.__checkpoint_stop = checkpoint_stop
        self.group = group
        self.training: bool
        self.transport = MakeTransport(
            use_rpc=("OMPI_COMM_WORLD_RANK" not in os.environ) or ("FORCE_RPC" in os.environ),
            worker_map=worker_map,
            input_device=input_device,
        )
        self.input_device = input_device
        self.final_stage = final_stage

    @property
    def checkpoint_stop(self) -> int:
        # Disable checkpointing if in eval mode.
        training = self.partitions[0].module.training
        if not training:
            return 0
        return self.__checkpoint_stop

    def run(self, training: bool, batches: List[Batch], event: Optional[Event]) -> None:

        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        self.training = training

        skip_trackers = [SkipTrackerThroughPotals(self.skip_layout, i) for i in range(len(batches))]

        rank = self.group.rank()
        event_loop = AsyncEventLoop(self.partitions, self.group, self.transport, self.training, self.checkpoint_stop,)
        if rank == 0 and not self.final_stage:
            logging.debug(f"{torch.distributed.get_rank()}: entered event head")
            event_loop.event_loop_head(batches, skip_trackers, event)
            logging.debug(f"{torch.distributed.get_rank()}: exited event head")
        elif self.final_stage:
            logging.debug(f"{torch.distributed.get_rank()}: entered event tail")
            event_loop.event_loop_tail(batches, skip_trackers)
            logging.debug(f"{torch.distributed.get_rank()}: exited event tail")
        else:
            logging.debug(f"{torch.distributed.get_rank()}: entered event loop")
            event_loop.event_loop(len(batches), skip_trackers)
            logging.debug(f"{torch.distributed.get_rank()}: exited event loop")

    def back_helper(self, output: List[Batch]) -> None:
        pass
