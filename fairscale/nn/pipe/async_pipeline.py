# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from threading import Event
from typing import List, Optional

import torch

from .async_schedule import AsyncEventLoop
from .microbatch import Batch
from .multiprocess_pipeline import MultiProcessPipeline
from .skip.tracker import SkipTrackerThroughPotals


class AsyncPipeline(MultiProcessPipeline):
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
