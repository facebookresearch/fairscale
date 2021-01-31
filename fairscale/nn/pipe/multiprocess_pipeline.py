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

"""The multiprocess pipeline parallelism of Pipe."""
import os
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event
from types import TracebackType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function

from fairscale.nn.model_parallel import get_pipeline_parallel_ranks

from .async_schedule import ModuleWrapper
from .checkpoint import Checkpointing
from .messages import MakeTransport, Transport
from .microbatch import Batch
from .skip import Namespace
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import ACTIVATIONS_GRADS_QUEUE, PORTAL_QUEUE, SKIP_TENSOR_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task

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
    def forward(ctx, dst_rank: int, tensor: Tensor, transport: Transport, index: int) -> Tensors:
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


def create_task(
    checkpoint_stop: int,
    i: int,
    j: int,
    batch: Batch,
    partition: nn.Sequential,
    skip_trackers: List[SkipTrackerThroughPotals],
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

        task = Task(None, compute=compute, finalize=None)
        del compute  # TODO(tom) maybe remove

    return task


class MultiProcessPipeline:
    """The multiprocess pipeline parallelism for Pipe."""

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

        m = len(batches)

        skip_trackers = [SkipTrackerThroughPotals(self.skip_layout, i) for i in range(m)]

        schedule = [(i, self.group.rank()) for i in range(m)]

        for i, j in schedule:
            assert len(self.partitions) == 1
            partition = self.partitions[0]

            if self.group.rank() != 0:
                batch = self.get_batch_from_previous_stage(i, skip_trackers, batches)
            else:
                batch = batches[i]

            task = create_task(self.checkpoint_stop, i, j, batch, partition.module, skip_trackers)

            batches[i] = self.execute_task(task, i, skip_trackers)

    def get_batch_from_previous_stage(
        self, i: int, skip_trackers: List[SkipTrackerThroughPotals], batches: List[Batch]
    ) -> Batch:

        phony = torch.empty(0, device=self.input_device, requires_grad=True)
        result = RecvOperator.apply(torch.distributed.get_rank(), phony, self.transport, i)
        if len(result) == 1:
            batch = Batch(result[0], i)
        else:
            batch = Batch(result, i)

        self.recv_skip_tensors(skip_trackers, batches)

        return batch

    def send_skip_tensors(
        self, this_rank: int, ranks: List[int], batch: Batch, i: int, skip_trackers: List[SkipTrackerThroughPotals]
    ) -> None:
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

        rank = self.group.rank()

        if not self.final_stage:
            ranks = get_pipeline_parallel_ranks()
            this_rank = torch.distributed.get_rank()

            self.send_skip_tensors(this_rank, ranks, batch, i, skip_trackers)
            SendOperator.apply(this_rank, ranks[ranks.index(this_rank) + 1], self.transport, [*batch], i)

        for portal in skip_trackers[i].portals.values():
            portal.pipeline = self

        task.finalize(batch)

        return batch

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
        tensors: Tensors

        rank = torch.distributed.get_rank()
        for batch in reversed(output):
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
