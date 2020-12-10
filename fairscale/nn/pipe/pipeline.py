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

"""The pipeline parallelism of Pipe."""
import logging
import os
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event
from types import TracebackType
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function

from fairscale.nn.model_parallel import get_pipeline_parallel_ranks

from .async_schedule import AsyncEventLoop, ModuleWrapper
from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .messages import MakeTransport, Transport
from .microbatch import Batch
from .skip import Namespace
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .types import (
    ACTIVATIONS_GRADS_QUEUE,
    PORTAL_QUEUE,
    SKIP_TENSOR_QUEUE,
    PipelineStyle,
    PipeMessage,
    Schedule,
    TensorOrTensors,
    Tensors,
)
from .worker import Task, create_workers, join_workers

__all__: List[str] = []

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


class SendOperator(torch.autograd.Function):
    """Send activations to the next pipeline stage"""

    @staticmethod
    # type: ignore
    def forward(ctx, src_rank, dst_rank, transport: Transport, input: List[Tensor], index: int) -> Tensors:
        assert src_rank == torch.distributed.get_rank()

        transport.send_message(
            PipeMessage(src_rank, dst_rank, queue_name=ACTIVATIONS_GRADS_QUEUE, args=index, tensors=tuple(input)),
        )
        return ()

    @staticmethod
    # type: ignore
    def backward(ctx, *grad: Tensor,) -> Tensors:
        return tuple(grad)


class RecvOperator(torch.autograd.Function):
    """Receive activations to the previous pipeline stage"""

    @staticmethod
    # type: ignore
    def forward(ctx, dst_rank: int, tensor: Tensor, input_device, transport: Transport, index: int) -> Tensors:
        assert dst_rank == torch.distributed.get_rank()
        ctx.transport = transport
        ctx.index = index

        result = transport.get_out_of_order(ACTIVATIONS_GRADS_QUEUE, index)

        def maybe_requires_grad(t: Tensor) -> Tensor:
            if t.dtype.is_floating_point:
                return t.requires_grad_()
            return t

        return tuple(maybe_requires_grad(r) for r in result)

    @staticmethod
    # type: ignore
    def backward(ctx, *grad: Tensor,) -> Tuple[Optional[Tensor], ...]:
        ranks = get_pipeline_parallel_ranks()
        this_rank = torch.distributed.get_rank()
        ctx.transport.send_message(
            PipeMessage(
                this_rank,
                ranks[ranks.index(this_rank) - 1],
                queue_name=ACTIVATIONS_GRADS_QUEUE,
                args=ctx.index,
                tensors=tuple(grad),
            ),
        )
        return (None, None, None, None, None)


# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional["Task"]]
    OutQueue = Queue[Tuple[bool, Union[Tuple["Task", Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def clock_cycles(m: int, n: int) -> Iterable[Schedule]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


def create_task(
    style: PipelineStyle,
    checkpoint_stop: int,
    i: int,
    j: int,
    batch: Batch,
    partition: nn.Sequential,
    skip_trackers: List[SkipTrackerThroughPotals],
    streams: List[AbstractStream],
) -> Task:
    # Determine whether checkpointing or not.
    if i < checkpoint_stop:

        def function(
            input: TensorOrTensors,
            partition: nn.Sequential = partition,
            skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
            chunk_id: int = i,
            part_id: int = j,
        ) -> TensorOrTensors:
            with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                ret = partition(input)
                # We do a check here because the backtrace from the checkpoint backward code path
                # is very hard to make sense. It would be much easier to check earlier at this point.
                assert type(ret) is not list, "Only Tensor or Tuple of Tensor output is supported"
                return ret

        chk = Checkpointing(function, batch)
        if style is PipelineStyle.SingleProcess:
            task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
        elif style in [PipelineStyle.MultiProcess, PipelineStyle.AsyncSchedule]:
            task = Task(None, compute=chk.checkpoint, finalize=chk.recompute)
        del function, chk  # TODO(tom) maybe remove

    else:

        def compute(
            batch: Batch = batch,
            partition: nn.Sequential = partition,
            skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
            chunk_id: int = i,
            part_id: int = j,
        ) -> Batch:
            with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                return batch.call(partition)

        if style is PipelineStyle.SingleProcess:
            task = Task(streams[j], compute=compute, finalize=None)
        elif style in [PipelineStyle.MultiProcess, PipelineStyle.AsyncSchedule]:
            task = Task(None, compute=compute, finalize=None)
        del compute  # TODO(tom) maybe remove

    return task


class Pipeline:
    """The pipeline parallelism for Pipe."""

    def __init__(
        self,
        partitions: List[nn.Sequential],
        devices: Optional[List[torch.device]],
        copy_streams: Optional[List[List[AbstractStream]]],
        skip_layout: SkipLayout,
        checkpoint_stop: int,
        style: PipelineStyle,
        group: Optional[torch.distributed.ProcessGroup] = None,
        worker_map: Optional[Dict[int, str]] = None,
        input_device: Union[None, int, str, torch.device] = None,
        final_stage: bool = False,
    ) -> None:
        if style == PipelineStyle.SingleProcess:
            self.partitions = partitions
        else:
            self.mp_partitions: List[ModuleWrapper] = cast(List[ModuleWrapper], partitions)
        self.devices = devices
        self.copy_streams = copy_streams
        self.skip_layout = skip_layout
        self.__checkpoint_stop = checkpoint_stop
        self.style = style
        self.group = group
        self.training: bool
        if style in [PipelineStyle.MultiProcess, PipelineStyle.AsyncSchedule]:
            self.transport = MakeTransport(
                use_rpc=("OMPI_COMM_WORLD_RANK" not in os.environ) or ("FORCE_RPC" in os.environ),
                worker_map=worker_map,
                input_device=input_device,
            )
        self.input_device = input_device
        self.all_at_once = False
        self.callcount = 0
        self.final_stage = final_stage
        if self.style is PipelineStyle.SingleProcess:
            assert self.devices is not None
            (self.in_queues, self.out_queues) = create_workers(self.devices)

    @property
    def checkpoint_stop(self) -> int:
        # Disable checkpointing if in eval mode.
        if self.style == PipelineStyle.SingleProcess:
            training = self.partitions[0].training
        else:
            training = self.mp_partitions[0].module.training
        if not training:
            return 0
        return self.__checkpoint_stop

    def __del__(self) -> None:
        if self.style is PipelineStyle.SingleProcess:
            join_workers(self.in_queues, self.out_queues)

    def run(self, training: bool, batches: List[Batch], event: Optional[Event]) -> None:

        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        self.training = training

        m = len(batches)

        skip_trackers = [SkipTrackerThroughPotals(self.skip_layout, i) for i in range(len(batches))]

        if self.style is PipelineStyle.SingleProcess:
            n = len(self.partitions)
            for schedule in clock_cycles(m, n):
                self.fence(batches, schedule, skip_trackers)
                self.compute(batches, schedule, skip_trackers)
        elif self.style is PipelineStyle.MultiProcess:
            assert self.group
            schedule = [(i, self.group.rank()) for i in range(m)]
            self.compute(batches, schedule, skip_trackers)
        elif self.style is PipelineStyle.AsyncSchedule:
            assert self.group
            rank = self.group.rank()
            event_loop = AsyncEventLoop(
                self.mp_partitions, self.group, self.transport, self.training, self.checkpoint_stop,
            )
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

        self.callcount += 1

    def fence(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        assert copy_streams
        assert skip_layout

        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0 and j != 0:
                depend(batches[i - 1], batches[i])

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j - 1][i]
                copy(batches[i], prev_stream, next_stream)

    def get_batch_from_previous_stage(
        self, i: int, skip_trackers: List[SkipTrackerThroughPotals], batches: List[Batch]
    ) -> Batch:

        phony = torch.empty(0, device=self.input_device, requires_grad=True)
        result = RecvOperator.apply(torch.distributed.get_rank(), phony, self.input_device, self.transport, i)
        if len(result) == 1:
            batch = Batch(result[0], i)
        else:
            batch = Batch(result, i)

        self.recv_skip_tensors(skip_trackers, batches)

        return batch

    def send_skip_tensors(
        self, this_rank: int, ranks: List[int], batch: Batch, i: int, skip_trackers: List[SkipTrackerThroughPotals]
    ) -> None:
        assert self.group
        for next_j, ns, name in self.skip_layout.copy_policy_by_src(self.group.rank()):
            life = skip_trackers[i].portals[(ns, name)].tensor_life
            loaded = skip_trackers[i].load(batch, ns, name)
            if loaded is not None:
                tensors = tuple([loaded])
            else:
                tensors = tuple()

            self.transport.send_message(
                PipeMessage(
                    this_rank, ranks[next_j], queue_name=SKIP_TENSOR_QUEUE, args=(i, ns, name, life), tensors=tensors,
                ),
                sync=True,
            )

    def recv_skip_tensors(self, skip_trackers: List[SkipTrackerThroughPotals], batches: List[Batch]) -> None:
        while True:
            try:
                message = self.transport.recv_message(SKIP_TENSOR_QUEUE, nowait=True)
                (si, ns, name, life) = message.args
                value: Optional[TensorOrTensors] = message.tensors

                assert isinstance(value, tuple)

                if len(value) == 0:
                    value = None
                else:
                    assert len(value) == 1
                    value = value[0]

                skip_trackers[si].save(batches[si], ns, name, value)
                old_life = skip_trackers[si].portals[(ns, name)].tensor_life
                if life != 0:
                    skip_trackers[si].portals[(ns, name)].tensor_life = life
            except QueueEmpty:
                break

    def execute_task(self, task: Task, i: int, skip_trackers: List[SkipTrackerThroughPotals]) -> Batch:
        batch = task.compute()

        assert self.group
        rank = self.group.rank()

        if self.style is PipelineStyle.MultiProcess and not self.final_stage:
            ranks = get_pipeline_parallel_ranks()
            this_rank = torch.distributed.get_rank()

            self.send_skip_tensors(this_rank, ranks, batch, i, skip_trackers)
            SendOperator.apply(this_rank, ranks[ranks.index(this_rank) + 1], self.transport, [*batch], i)

        for portal in skip_trackers[i].portals.values():
            portal.pipeline = self

        task.finalize(batch)

        return batch

    def finalize_tasks(
        self,
        n: int,
        schedule: Schedule,
        streams: List[AbstractStream],
        copy_streams: List[List[AbstractStream]],
        batches: List[Batch],
    ) -> None:
        exc_info: Optional[ExcInfo] = None
        for i, j in schedule:
            ok, payload = self.out_queues[j].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n - 1:
                wait(batch, streams[j], copy_streams[j][i])

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            assert self.devices
            with use_device(self.devices[j]):
                task.finalize(batch)

            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    def compute(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals]
    ) -> None:
        """Runs tasks with synchronization to copy streams."""
        devices = self.devices
        copy_streams = self.copy_streams

        if self.style is PipelineStyle.SingleProcess:
            assert devices is not None
            n = len(self.partitions)
            streams = [current_stream(d) for d in devices]
        elif self.style is PipelineStyle.MultiProcess:
            assert self.group
            n = self.group.size()
            streams = []

        # With checkpointing, the autograd graph looks like this diagram:
        # ┌─────┸──────┐
        # │    Copy    │
        # └─────┰──────┘   (fence)
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #       ┃          (compute)
        # ┌─────┸──────┐
        # │    Wait    │ [1] Synchronize the current stream with the copy stream.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │ Checkpoint │ [2] Compute a partition within checkpointing.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │    Wait    │ [3] Synchronize the copy stream with the current stream.
        # └─────┰──────┘
        #       ┠ ─ ─ ─ ┐
        #       ┃ ┌─────┴─────┐
        #       ┃ │ Recompute │ [4] Schedule the recomputation at backpropagation.
        #       ┃ └─────┬─────┘
        #       ┠ ─ ─ ─ ┘
        #       ┃
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # ┌─────┸──────┐   (fence)
        # │    Copy    │
        # └─────┰──────┘
        for i, j in schedule:
            batch = batches[i]

            if self.style is PipelineStyle.SingleProcess:
                partition = self.partitions[j]
                # Synchronize with the copied input. ([1] in the diagram)
                assert copy_streams
                if j != 0:
                    wait(batch, copy_streams[j][i], streams[j])

                task = create_task(self.style, self.checkpoint_stop, i, j, batch, partition, skip_trackers, streams)

                # Compute tasks in parallel. ([2] in the diagram)
                self.in_queues[j].put(task)
            elif self.style is PipelineStyle.MultiProcess:
                assert len(self.mp_partitions) == 1
                mp_partition = self.mp_partitions[0]

                assert self.group
                if self.group.rank() != 0:
                    batch = self.get_batch_from_previous_stage(i, skip_trackers, batches)

                task = create_task(
                    self.style, self.checkpoint_stop, i, j, batch, mp_partition.module, skip_trackers, streams
                )

                batches[i] = self.execute_task(task, i, skip_trackers)

        if self.style is PipelineStyle.SingleProcess:
            assert copy_streams
            self.finalize_tasks(n, schedule, streams, copy_streams, batches)

    def send_portal_grad(self, ns_name: Tuple[Namespace, str], index: int, grad: TensorOrTensors) -> None:
        dest, src = self.skip_layout.by_ns_name.get(ns_name, (-1, -1))
        if dest == src:
            return
        ranks = get_pipeline_parallel_ranks()
        dst_rank = ranks[dest]
        if dst_rank == torch.distributed.get_rank():
            return

        if isinstance(grad, Tensor):
            grad = tuple([grad])
        self.transport.send_message(
            PipeMessage(ranks[src], dst_rank, queue_name=PORTAL_QUEUE, args=(ns_name, index), tensors=grad), sync=True,
        )

    def recv_portal_grad(self, expected_ns_name: Tuple[Namespace, str], expected_index: int) -> Tensor:
        message = self.transport.recv_message(PORTAL_QUEUE)

        (ns_name, index) = message.args
        grad = message.tensors

        assert len(grad) == 1
        result = grad[0]
        assert index == expected_index and ns_name == expected_ns_name
        return result

    def back_helper(self, output: List[Batch]) -> None:
        if self.style == PipelineStyle.AsyncSchedule:
            return

        o = list(output)

        tensors: Tensors

        if self.all_at_once:
            # FIXME(tom) allow specifying this branch when constructing Pipe(), add a test
            grads = []
            for i, batch in enumerate(o):
                rank = torch.distributed.get_rank()
                found = self.transport.get_out_of_order(ACTIVATIONS_GRADS_QUEUE, i)
                assert len(found) == 1
                grads.append(found[0])
                tensors = tuple(x.tensor_or_tensors for x in o)  # type: ignore
            try:
                torch.autograd.backward(tensors, grad_tensors=grads, retain_graph=True)
            except Exception as e:
                raise RuntimeError("Autograd failed") from e
        else:
            rank = torch.distributed.get_rank()
            for batch in o:
                found = self.transport.get_out_of_order(ACTIVATIONS_GRADS_QUEUE, batch.index)
                if batch.atomic:
                    tensors = tuple([batch.tensor])
                else:
                    tensors = batch.tensors

                if len(found) != len(tensors):
                    raise RuntimeError("different number of tensors and gradients")

                grads = []
                final_tensors = []
                for i, tensor in enumerate(tensors):
                    if tensor.requires_grad or getattr(tensor, "grad_fn", None) is not None:
                        grads.append(found[i])
                        final_tensors.append(tensor)

                try:
                    torch.autograd.backward(final_tensors, grad_tensors=grads, retain_graph=True)
                except Exception as e:
                    raise RuntimeError(f"Autograd failed on {torch.distributed.get_rank()}") from e
