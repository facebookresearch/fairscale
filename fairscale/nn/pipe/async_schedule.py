# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup

from fairscale.nn.model_parallel import get_pipeline_parallel_ranks

from .checkpoint import Checkpointing
from .messages import Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task


def create_task(
    checkpoint_stop: int,
    chunk_id: int,
    part_id: int,
    batch: Batch,
    partition: nn.Sequential,
    skip_trackers: List[SkipTrackerThroughPotals],
) -> Task:
    # Determine whether checkpointing or not.
    if chunk_id < checkpoint_stop:

        def function(
            input: TensorOrTensors,
            partition: nn.Sequential = partition,
            skip_tracker: SkipTrackerThroughPotals = skip_trackers[chunk_id],
            chunk_id: int = chunk_id,
            part_id: int = part_id,
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
            skip_tracker: SkipTrackerThroughPotals = skip_trackers[chunk_id],
            chunk_id: int = chunk_id,
            part_id: int = part_id,
        ) -> Batch:
            with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                return batch.call(partition)

        task = Task(None, compute=compute, finalize=None)
        del compute  # TODO(tom) maybe remove

    return task


@dataclass(frozen=True)
class Location:
    stage: int
    index: int

    def __repr__(self) -> str:
        return f"{self.stage}@{self.index}"


@dataclass(frozen=True)
class Invocation:
    order: int
    this: Location
    source: Optional[Location]
    dest: Optional[Location]


Activations = Dict[int, Dict[int, Dict[int, Batch]]]
Invocations = Dict[int, Invocation]


@dataclass(frozen=True)
class TailBackwardContext:
    activations: Activations
    invocations: Invocations
    count_per_order: Dict[int, int]
    expected_gradients: int


class ModuleWrapper:
    def __init__(self, module: nn.Sequential, location: Location, invocations: Optional[List[Invocation]] = None):
        self.module: nn.Sequential = module
        self.location: Location = location
        self.invocations: List[Invocation] = invocations or []

    def __repr__(self) -> str:
        return f"{self.location}:\n" + "\n".join(map(str, self.invocations)) + "\n\t" + str(self.module)

    def __len__(self) -> int:
        return len(self.module)

    def __iter__(self) -> Iterable:
        yield from self.module


class AsyncMessageType(Enum):
    Activations = auto()
    Gradients = auto()


@dataclass(frozen=True)
class AsyncMessageBody:
    message_type: AsyncMessageType
    microbatch_index: int
    source: Location
    dest: Location
    order: int


class AutogradWithoutActivations(torch.autograd.Function):
    """A helper class to add another edge in the autograd graph which allows us
    to delete the potentially large activations and still perform a backward
    pass. Returns return a phony tensor which is connected to the graph."""

    @staticmethod
    # type: ignore
    def forward(ctx, *x):
        return torch.tensor(1.0)

    @staticmethod
    # type: ignore
    def backward(ctx, grad):
        assert ctx.grad_from_pipeline is not None
        return ctx.grad_from_pipeline


class AsyncRecvOperator(torch.autograd.Function):
    """Receive activations to the previous pipeline stage"""

    @staticmethod
    # type: ignore
    def forward(ctx, phony: Tensor, transport: Transport, message: PipeMessage, queue_name: int) -> Tensors:
        ctx.transport = transport
        ctx.index = message.args.microbatch_index
        ctx.queue_name = queue_name
        result = transport.recv_message_tensors(message)

        ctx.args = result.args

        def maybe_requires_grad(t: Tensor) -> Tensor:
            if t.dtype.is_floating_point:
                return t.requires_grad_()
            return t

        return tuple(maybe_requires_grad(r) for r in result.tensors)

    @staticmethod
    # type: ignore
    def backward(ctx, *grad: Tensor,) -> Tuple[Optional[Tensor], ...]:
        ranks = get_pipeline_parallel_ranks()
        this_rank = torch.distributed.get_rank()
        body = AsyncMessageBody(
            AsyncMessageType.Gradients, ctx.index, source=ctx.args.dest, dest=ctx.args.source, order=ctx.args.order - 1
        )
        ctx.transport.send_message(
            PipeMessage(
                this_rank, ranks[ctx.args.source.stage], queue_name=ctx.queue_name, args=body, tensors=tuple(grad),
            ),
            sync=True,
        )

        tail_ctx = getattr(ctx, "tail_ctx", None)
        if tail_ctx:
            expected_gradients = tail_ctx.expected_gradients
            while expected_gradients > 0:
                message = ctx.transport.recv_message_header(ctx.queue_name)

                args: AsyncMessageBody = message.args
                assert args.message_type is AsyncMessageType.Gradients

                invocation = tail_ctx.invocations[args.order]
                expected_gradients -= tail_ctx.count_per_order[invocation.order]
                AsyncEventLoop.perform_backward_for_invocation(ctx.transport, message, tail_ctx.activations, invocation)

        return (None, None, None, None, None)


class AsyncEventLoop:
    def __init__(
        self,
        partitions: List[ModuleWrapper],
        group: ProcessGroup,
        transport: Transport,
        training: bool,
        checkpoint_stop: int,
    ):
        self.training = training
        self.checkpoint_stop = checkpoint_stop
        self.transport = transport
        self.group = group
        self.partitions: List[ModuleWrapper] = partitions

    def send_async_message(self, dst_rank: int, result: Batch, invocation: Invocation) -> Batch:
        """Send batch to dst_rank, and use AutogradWithoutActivations to delete
        the activations since we no longer need them"""

        assert invocation.dest
        src_rank = torch.distributed.get_rank()

        body = AsyncMessageBody(
            AsyncMessageType.Activations, result.index, invocation.this, invocation.dest, invocation.order + 1
        )
        self.transport.send_message(
            PipeMessage(src_rank, dst_rank, queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple([*result])),
            sync=True,
        )

        phony = AutogradWithoutActivations.apply(*result)
        return Batch(phony, result.index)

    def run_invocation(
        self,
        batch: Batch,
        partition: ModuleWrapper,
        skip_trackers: List[SkipTrackerThroughPotals],
        invocation: Invocation,
    ) -> Batch:
        """Actually run the forward pass for a given module, and send the result
        to the next stage in the pipeline if needed."""

        task = create_task(
            self.checkpoint_stop, batch.index, self.group.rank(), batch, partition.module, skip_trackers,
        )
        result = task.compute()
        task.finalize(result)

        if invocation.dest and invocation.dest.stage != invocation.this.stage:
            ranks = get_pipeline_parallel_ranks()
            dst_rank = ranks[invocation.dest.stage]
            result = self.send_async_message(dst_rank, result, invocation)
        return result

    @staticmethod
    def perform_backward_for_invocation(
        transport: Transport, message: PipeMessage, activations: Activations, invocation: Invocation
    ) -> None:
        """Perform the backward pass by looking up the appropriate `Batch` and
        then calling `backward` on the tensor"""

        recvd_grads = transport.recv_message_tensors(message)

        batch: Batch = activations[invocation.this.index][invocation.order][message.args.microbatch_index]

        # All batches saved in `activations` are generated by AutogradWithoutActivations,
        # so we store the gradients in `grad_from_pipeline` so it will be used
        # during the backward pass
        batch.tensor.grad_fn.grad_from_pipeline = tuple(recvd_grads.tensors)  # type: ignore
        batch.tensor.backward(retain_graph=True)

    def run_invocations_on_batch(
        self,
        batch: Batch,
        invocations: Invocations,
        order: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
    ) -> Tuple[int, int]:
        """Run invocations on the batch until we hit one that receives its input
        from a different stage (i.e. another process)"""

        invocations_handled = 0
        last_order = 0
        for invocation in invocations.values():
            if invocation.order < order:
                continue
            pi = invocation.this.index
            partition = self.partitions[pi]

            if invocation.order == order:
                invocations_handled += 1
                last_order = invocation.order
                activations[pi][invocation.order][batch.index] = self.run_invocation(
                    batch, partition, skip_trackers, invocation
                )
            elif invocation.source and invocation.source.stage == self.group.rank():
                invocations_handled += 1
                last_order = invocation.order
                batch = activations[invocation.source.index][invocation.order - 1][batch.index]
                activations[pi][invocation.order][batch.index] = self.run_invocation(
                    batch, partition, skip_trackers, invocation
                )
                del activations[invocation.source.index][invocation.order - 1][batch.index]

            elif invocation.source and invocation.source.stage != self.group.rank():
                break

        return (invocations_handled, last_order)

    def event_loop_head(
        self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals], event: Optional[Event]
    ) -> None:
        """The event loop for the "head", which first performs the forward pass
        on any applicable layers for this stage, and then enters the common
        `event_loop_inner`"""

        invocations, activations = self.get_invocations_and_activations()

        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0

        count_per_order = dict()

        for batch in batches:
            inv_count, last_order = self.run_invocations_on_batch(batch, invocations, 0, skip_trackers, activations)
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count

        if actual_invocations < expected_invocations or self.training:
            self.event_loop_inner(
                expected_invocations,
                skip_trackers,
                activations,
                invocations,
                count_per_order,
                already_received=actual_invocations,
                event=event,
            )

    def get_batch_from_message(self, message: PipeMessage) -> Batch:
        """Get the tensor(s) wrapped in a `Batch` from a `PipeMessage`, applying
        AsyncRecvOperator so we can intercept the backward pass"""

        microbatch_index = message.args.microbatch_index
        phony = torch.empty(0, device=self.transport.input_device, requires_grad=True)
        result = AsyncRecvOperator.apply(phony, self.transport, message, EVENT_LOOP_QUEUE)
        if len(result) == 1:
            batch = Batch(result[0], microbatch_index)
        else:
            batch = Batch(result, microbatch_index)
        return batch

    def event_loop_tail(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "tail", or final stage which only processes
        activations and then returns to the caller so that the loss can be
        calculated. This also handles the first/only stage for the special
        case of a 1-stage pipeline."""

        invocations, activations = self.get_invocations_and_activations()
        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0

        rank = self.group.rank()
        count_per_order = dict()

        for batch in batches:
            if rank == 0:
                order = 0
            else:
                message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
                args: AsyncMessageBody = message.args

                batch = self.get_batch_from_message(message)
                order = args.order

            inv_count, last_order = self.run_invocations_on_batch(batch, invocations, order, skip_trackers, activations)
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count

            if invocations[last_order].dest is None:
                self.prepare_tail_backward(
                    batch, activations, invocations, count_per_order, len(invocations) - inv_count
                )

        if actual_invocations < expected_invocations:
            expected_gradients = 0  # (len(invocations) - 1) * len(batches)

            self.event_loop_inner(
                expected_invocations,
                skip_trackers,
                activations,
                invocations,
                count_per_order,
                already_received=actual_invocations,
                ignore_gradients=True,
                tail=True,
            )

        _, last_invocation = invocations.popitem()

        for index, batch in activations[len(self.partitions) - 1][last_invocation.order].items():
            batches[index] = batch

    def get_invocations_and_activations(self) -> Tuple[Invocations, Activations]:
        activations: Activations = dict()
        invocations: Invocations = OrderedDict()

        for pi, partition in enumerate(self.partitions):
            activations[pi] = dict()
            for invocation in partition.invocations:
                activations[pi][invocation.order] = dict()
                invocations[invocation.order] = invocation

        invocations = OrderedDict(sorted(invocations.items(), key=lambda entry: entry[0]))

        return (invocations, activations)

    def event_loop(self, num_microbatch: int, skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "middle", i.e. neither the head nor the tail"""

        invocations, activations = self.get_invocations_and_activations()

        expected_invocations = len(invocations) * num_microbatch

        self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, dict())

    def event_loop_inner(
        self,
        expected_invocations: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
        invocations: Invocations,
        count_per_order: Dict[int, int],
        *,
        already_received: int = 0,
        ignore_gradients: bool = False,
        event: Optional[Event] = None,
        tail: bool = False,
    ) -> None:
        """The common event loop shared by all stages. This processses
        activations for the forward pass, and if `self.training` is true,
        processes gradients for the backward pass."""

        num_activations = already_received
        if self.training and not ignore_gradients:
            num_gradients = 0
        else:
            num_gradients = expected_invocations

        while num_activations < expected_invocations or num_gradients < expected_invocations:
            if num_activations == expected_invocations and num_gradients == 0 and event is not None:
                # We are ready to do the backward pass, but must wait for
                # PipeRPCWrapper to signal that it is safe to proceed, otherwise
                # deadlock
                event.wait()

            message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
            args: AsyncMessageBody = message.args

            invocation = invocations[args.order]

            # FIXME(tom) for combining pipeline with megatron, I currently don't
            # control the order of received activations or gradients, so it is
            # possible for a reused ColumnParallelLinear for example to receive
            # a different order of activations w.r.t. the sending stage, which
            # would result in incorrect values being used for the all_gather
            if args.message_type is AsyncMessageType.Activations:
                batch = self.get_batch_from_message(message)

                inv_count, last_order = self.run_invocations_on_batch(
                    batch, invocations, args.order, skip_trackers, activations
                )
                count_per_order[last_order] = inv_count
                num_activations += inv_count
                if tail and invocations[last_order].dest is None:
                    self.prepare_tail_backward(
                        batch, activations, invocations, count_per_order, len(invocations) - inv_count
                    )

                assert num_activations <= expected_invocations

            elif args.message_type is AsyncMessageType.Gradients:
                num_gradients += count_per_order[invocation.order]
                self.perform_backward_for_invocation(self.transport, message, activations, invocation)

    @staticmethod
    def prepare_tail_backward(
        batch: Batch,
        activations: Activations,
        invocations: Invocations,
        count_per_order: Dict[int, int],
        expected_gradients: int,
    ) -> None:
        if expected_gradients > 0:
            grad_fn = next(b.grad_fn for b in batch if b.requires_grad)
            assert grad_fn
            grad_fn.tail_ctx = TailBackwardContext(activations, invocations, count_per_order, expected_gradients)
