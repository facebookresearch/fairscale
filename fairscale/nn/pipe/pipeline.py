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
from enum import Enum, auto
import os
import pickle
from queue import Empty as QueueEmpty
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function

from fairscale.nn.model_parallel import get_pipeline_parallel_ranks

from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .microbatch import Batch
from .skip import Namespace
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .worker import Task, create_workers, join_workers

__all__: List[str] = []


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

InputDevice = Union[None, int, str, torch.device]
Schedule = List[Tuple[int, int]]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

MessageQueues: List[Queue] = [Queue(), Queue(), Queue()]
ACTIVATIONS_GRADS_QUEUE = 0
SKIP_TENSOR_QUEUE = 1
PORTAL_QUEUE = 2
MESSAGE_GENERATION_START = 3


# FIXME Why is 256 ok for training but not for tests?
MESSAGE_TENSOR_SIZE = 512  # 256

MessageGeneration = MESSAGE_GENERATION_START


class PipelineStyle(Enum):
    SingleProcess = auto()
    MultiProcess = auto()


@dataclass(frozen=True)
class TransportConfig:
    use_rpc: bool
    worker_map: Optional[Dict[int, str]]


@dataclass
class PipeMessage:
    src: int
    dest: int
    queue_name: int
    args: Any
    tensors: Tensors
    tensor_shapes: List[torch.Size]
    tensor_dtypes: List[torch.dtype]
    tag: int = 0

    def __init__(self, src: int, dest: int, queue_name: int, args: Any, tensors: Tensors):
        self.src = src
        self.dest = dest
        self.queue_name = queue_name
        self.args = args
        self.tensors = tensors

        global MessageGeneration
        self.tag = MessageGeneration
        MessageGeneration += len(tensors)


def rpc_push_queue(message: PipeMessage) -> None:
    globals()["MessageQueues"][message.queue_name].put(message)


def pyobject_to_tensor(obj: Any) -> Tensor:
    pickled = pickle.dumps(obj)
    nparray = np.frombuffer(pickled, dtype=np.uint8).copy()
    nparray.setflags(write=True)
    result = torch.from_numpy(nparray)
    delta = MESSAGE_TENSOR_SIZE - len(result)
    if delta < 0:
        raise ValueError(
            f"message too big to send, increase MESSAGE_TENSOR_SIZE? - {len(result)} > {MESSAGE_TENSOR_SIZE}"
        )
    elif delta > 0:
        result = torch.cat((result, torch.zeros(delta, dtype=torch.uint8)))

    return result.cuda()


def tensor_to_pyobject(tensor: Tensor) -> Any:
    nparray = tensor.numpy()
    return pickle.loads(nparray.tobytes())


def send_message(config: TransportConfig, message: PipeMessage, sync: bool = False) -> None:
    if config.use_rpc:
        message.tensors = tuple(t.cpu() for t in message.tensors)
        assert config.worker_map
        name = config.worker_map[message.dest]
        if sync:
            torch.distributed.rpc.rpc_sync(name, rpc_push_queue, args=(message,))
        else:
            torch.distributed.rpc.rpc_async(name, rpc_push_queue, args=(message,))
    else:
        tensors = message.tensors
        message.tensors = tuple()
        message.tensor_shapes = [t.size() for t in tensors]
        message.tensor_dtypes = [t.dtype for t in tensors]
        torch.cuda.current_stream().synchronize()
        torch.distributed.send(pyobject_to_tensor(message), message.dest, tag=0)
        for index, t in enumerate(tensors):
            if t.device.type == "cpu":
                t = t.cuda()
            torch.distributed.send(t, message.dest, tag=message.tag + index)


def recv_message(
    config: TransportConfig, queue_name: int, *, nowait: bool = False, input_device: InputDevice = None
) -> PipeMessage:
    if config.use_rpc:
        queue = globals()["MessageQueues"][queue_name]
        if nowait:
            result = queue.get_nowait()
        else:
            result = queue.get()
        result.tensors = to_input_device(result.tensors, input_device)
        return result
    else:
        # FIXME(handle nowait)
        if nowait:
            raise QueueEmpty

        tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=input_device)
        torch.distributed.recv(tensor, src=-1, tag=queue_name)
        message = tensor_to_pyobject(tensor.cpu())

        torch.cuda.current_stream().synchronize()

        message_tensors = []
        for index, (shape, dtype) in enumerate(zip(message.tensor_shapes, message.tensor_dtypes)):
            t = torch.empty(*shape, dtype=dtype, device=input_device)
            torch.distributed.recv(t, message.src, tag=message.tag + index)
            message_tensors.append(t)

        message.tensors = tuple(message_tensors)

        torch.cuda.current_stream().synchronize()
        return message


def get_out_of_order(config: TransportConfig, queue_name: int, index: int, *, input_device: InputDevice) -> Tensors:
    """Receive a message with a known microbatch index, and handle out-of-order
    messages by placing them back on the queue"""

    if config.use_rpc:
        queue = globals()["MessageQueues"][queue_name]
        out_of_order: List[PipeMessage] = []
        while True:
            message = recv_message(config, queue_name, input_device=input_device)
            got_index = message.args
            value = message.tensors
            if got_index == index:
                for b in out_of_order:
                    queue.put(b)
                return value
            else:
                out_of_order.append(message)
    else:
        message = recv_message(config, queue_name, input_device=input_device)
        assert message.args == index
        return message.tensors


def to_input_device(tensors: TensorOrTensors, input_device: InputDevice) -> TensorOrTensors:
    if input_device is None:
        return tensors
    else:
        if isinstance(tensors, Tensor):
            return tensors.to(input_device)
        else:
            return tuple(t.to(input_device) for t in tensors)


class SendOperator(torch.autograd.Function):
    """Send activations to the next pipeline stage"""

    @staticmethod
    # type: ignore
    def forward(ctx, src_rank, dst_rank, config: TransportConfig, input: List[Tensor], index: int) -> Tensors:
        assert src_rank == torch.distributed.get_rank()

        send_message(
            config,
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
    def forward(ctx, dst_rank: int, tensor: Tensor, input_device, config: TransportConfig, index: int) -> Tensors:
        assert dst_rank == torch.distributed.get_rank()
        ctx.config = config
        ctx.index = index

        result = get_out_of_order(config, ACTIVATIONS_GRADS_QUEUE, index, input_device=input_device)

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
        send_message(
            ctx.config,
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
    ) -> None:
        self.partitions = partitions
        self.devices = devices
        self.copy_streams = copy_streams
        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop
        self.style = style
        self.group = group
        self.transport_config = TransportConfig(
            use_rpc=("OMPI_COMM_WORLD_RANK" not in os.environ), worker_map=worker_map
        )

        self.input_device = input_device
        self.all_at_once = False
        self.callcount = 0
        if self.style is PipelineStyle.SingleProcess:
            assert self.devices is not None
            (self.in_queues, self.out_queues) = create_workers(self.devices)

    def __del__(self) -> None:
        if self.style is PipelineStyle.SingleProcess:
            join_workers(self.in_queues, self.out_queues)

    def run(self, batches: List[Batch]) -> None:

        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        partitions = self.partitions
        devices = self.devices

        m = len(batches)
        n = len(partitions)

        skip_trackers = [SkipTrackerThroughPotals(self.skip_layout, i) for i in range(len(batches))]

        if self.style is PipelineStyle.SingleProcess:
            for schedule in clock_cycles(m, n):
                self.fence(batches, schedule, skip_trackers)
                self.compute(batches, schedule, skip_trackers)
        elif self.style is PipelineStyle.MultiProcess:
            assert self.group
            schedule = [(i, self.group.rank()) for i in range(m)]
            self.compute(batches, schedule, skip_trackers)

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
        result = RecvOperator.apply(torch.distributed.get_rank(), phony, self.input_device, self.transport_config, i)
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

            send_message(
                self.transport_config,
                PipeMessage(
                    this_rank, ranks[next_j], queue_name=SKIP_TENSOR_QUEUE, args=(i, ns, name, life), tensors=tensors,
                ),
                sync=True,
            )

    def recv_skip_tensors(self, skip_trackers: List[SkipTrackerThroughPotals], batches: List[Batch]) -> None:
        while True:
            try:
                message = recv_message(
                    self.transport_config, SKIP_TENSOR_QUEUE, nowait=True, input_device=self.input_device
                )
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

        if rank != self.group.size() - 1:
            ranks = get_pipeline_parallel_ranks()
            this_rank = torch.distributed.get_rank()

            self.send_skip_tensors(this_rank, ranks, batch, i, skip_trackers)
            SendOperator.apply(this_rank, ranks[ranks.index(this_rank) + 1], self.transport_config, [*batch], i)

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

    def create_task(
        self,
        i: int,
        j: int,
        batch: Batch,
        checkpoint_stop: int,
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
                    return partition(input)

            chk = Checkpointing(function, batch)
            if self.style is PipelineStyle.SingleProcess:
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
            elif self.style is PipelineStyle.MultiProcess:
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

            if self.style is PipelineStyle.SingleProcess:
                task = Task(streams[j], compute=compute, finalize=None)
            elif self.style is PipelineStyle.MultiProcess:
                task = Task(None, compute=compute, finalize=None)
            del compute  # TODO(tom) maybe remove

        return task

    def compute(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Runs tasks with synchronization to copy streams."""
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        # Disable checkpointing if in eval mode.
        if not self.partitions[0].training:
            checkpoint_stop = 0

        if self.style is PipelineStyle.SingleProcess:
            assert devices is not None
            n = len(partitions)
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
                partition = partitions[j]
                # Synchronize with the copied input. ([1] in the diagram)
                assert copy_streams
                if j != 0:
                    wait(batch, copy_streams[j][i], streams[j])
            elif self.style is PipelineStyle.MultiProcess:
                assert len(self.partitions) == 1
                partition = self.partitions[0]

                assert self.group
                if self.group.rank() != 0:
                    batch = self.get_batch_from_previous_stage(i, skip_trackers, batches)

            task = self.create_task(i, j, batch, checkpoint_stop, partition, skip_trackers, streams)

            if self.style is PipelineStyle.SingleProcess:
                # Compute tasks in parallel. ([2] in the diagram)
                self.in_queues[j].put(task)
            elif self.style is PipelineStyle.MultiProcess:
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
        send_message(
            self.transport_config,
            PipeMessage(ranks[src], dst_rank, queue_name=PORTAL_QUEUE, args=(ns_name, index), tensors=grad),
            sync=True,
        )

    def recv_portal_grad(self, expected_ns_name: Tuple[Namespace, str], expected_index: int) -> Tensor:
        message = recv_message(self.transport_config, PORTAL_QUEUE, input_device=self.input_device)

        (ns_name, index) = message.args
        grad = message.tensors

        assert len(grad) == 1
        result = grad[0]
        assert index == expected_index and ns_name == expected_ns_name
        return result

    def back_helper(self, output: List[Batch]) -> None:
        o = list(output)

        tensors: Tensors

        if self.all_at_once:
            # FIXME(tom) allow specifying this branch when constructing Pipe(), add a test
            grads = []
            for i, batch in enumerate(o):
                rank = torch.distributed.get_rank()
                found = get_out_of_order(
                    self.transport_config, ACTIVATIONS_GRADS_QUEUE, i, input_device=self.input_device
                )
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
                found = get_out_of_order(
                    self.transport_config, ACTIVATIONS_GRADS_QUEUE, batch.index, input_device=self.input_device
                )
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
                    raise RuntimeError("Autograd failed") from e
