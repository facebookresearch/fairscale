# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Tuple, Callable, Any

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from fairscale.nn.model_parallel import (
    get_model_parallel_group,
    get_model_parallel_group_with_pipeline_backend,
    get_pipeline_parallel_ranks,
)

from .messages import IRecvWrapper, Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals
from .types import EVENT_LOOP_QUEUE, PipelineStyle, PipeMessage, Tensors, TensorOrTensors


@dataclass(frozen=True)
class Location:
    # Pipeline stage or rank where the module is located
    stage: int
    # Position of the module within a stage
    index: int

    def __repr__(self) -> str:
        return f"{self.stage}@{self.index}"


@dataclass(frozen=True)
class Invocation:
    # Position in the entire pipeline
    order: int
    # Location of the module for this Invocation
    this: Location
    # Location of the source module for this Invocation, may be None if this is
    # the very first Invocation, i.e. order == 0
    source: Optional[Location]
    # Location of the destination module for this Invocation, may be None if this is
    # the very last Invocation, i.e. order == maximum order seen throughout the
    # pipeline
    dest: Optional[Location]

    def sends_activation(self) -> bool:
        return bool(self.dest and self.dest.stage != self.this.stage)

    def receives_activation(self) -> bool:
        return bool(self.source and self.source.stage != self.this.stage)


Activations = Dict[int, Dict[int, Batch]]
Invocations = Dict[int, Invocation]


@dataclass(frozen=True)
class BackwardContext:
    activations: Activations
    invocations: Invocations
    count_per_order: Dict[int, int]
    num_batches: Optional[int] = None
    expected_recv: Optional[int] = None


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
    def forward(ctx, phony: Tensor, transport: Transport, message: PipeMessage) -> Tensors:
        ctx.transport = transport
        ctx.index = message.args.microbatch_index

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
                this_rank, ranks[ctx.args.source.stage], queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple(grad),
            ),
            sync=True,
        )

        tail_ctx = getattr(ctx, "tail_ctx", None)

        if tail_ctx:
            expected_gradients = tail_ctx.expected_recv
            assert expected_gradients is not None
            for _ in range(expected_gradients):
                message = ctx.transport.recv_message_header(EVENT_LOOP_QUEUE)
                args: AsyncMessageBody = message.args

                assert args.message_type is AsyncMessageType.Gradients

                invocation = tail_ctx.invocations[args.order]

                AsyncEventLoop.perform_backward_for_invocation(ctx.transport, message, tail_ctx.activations, invocation)

            ctx.tail_ctx = None
            del tail_ctx

        return (None, None, None, None, None)


class AsyncEventLoop:
    def __init__(
        self,
        partitions: List[ModuleWrapper],
        group: ProcessGroup,
        transport: Transport,
        training: bool,
        checkpoint_stop: int,
        loss_func: Optional[Callable[[TensorOrTensors, TensorOrTensors], TensorOrTensors]] = None,
    ):
        self.training = training
        self.checkpoint_stop = checkpoint_stop
        self.transport = transport
        self.group = group
        self.partitions: List[ModuleWrapper] = partitions
        self.loss_func = loss_func

    def send_async_message(self, dst_rank: int, result: Batch, invocation: Invocation) -> Batch:
        """Send batch to dst_rank, and use AutogradWithoutActivations to delete
        the activations since we no longer need them"""

        assert invocation.dest
        src_rank = torch.distributed.get_rank()

        body = AsyncMessageBody(
            AsyncMessageType.Activations,
            result.index,
            source=invocation.this,
            dest=invocation.dest,
            order=invocation.order + 1,
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
        target: Optional[List[Batch]] = None,
    ) -> Batch:
        """Actually run the forward pass for a given module, and send the result
        to the next stage in the pipeline if needed."""
        assert self.group
        from .pipeline import create_task

        target_args: Dict[str, Any] = {}
        if self.training and target is not None and self.loss_func is not None and invocation.dest is None:
            target_args["target"] = target.pop(0)
            target_args["loss_func"] = self.loss_func

        does_send = (invocation.dest is not None) and invocation.dest.stage != invocation.this.stage
        should_split = does_send and batch.index < self.checkpoint_stop

        task = create_task(
            PipelineStyle.AsyncSchedule,
            self.checkpoint_stop,
            batch.index,
            self.group.rank(),
            batch,
            partition.module,
            skip_trackers,
            streams=[],
            should_split=should_split,
            **target_args,
        )
        result = task.compute()

        if should_split:
            split_tensor = task.split()
        else:
            split_tensor = None
            task.finalize(result)

        if does_send:
            assert invocation.dest is not None
            ranks = get_pipeline_parallel_ranks()
            dst_rank = ranks[invocation.dest.stage]
            result = self.send_async_message(dst_rank, result, invocation)

        result.split = split_tensor

        return result

    @staticmethod
    def perform_backward_for_invocation(
        transport: Transport, message: PipeMessage, activations: Activations, invocation: Invocation
    ) -> None:
        """Perform the backward pass by looking up the appropriate `Batch` and
        then calling `backward` on the tensor"""

        recvd_grads = transport.recv_message_tensors(message)

        batch: Batch = activations[invocation.order][message.args.microbatch_index]

        # All batches saved in `activations` are generated by AutogradWithoutActivations,
        # so we store the gradients in `grad_from_pipeline` so it will be used
        # during the backward pass
        batch.tensor.grad_fn.grad_from_pipeline = tuple(recvd_grads.tensors)  # type: ignore
        # torch.autograd.backward(batch.tensor_or_tensors, grad_tensors=recvd_grads.tensors, retain_graph=True)
        batch.tensor.backward(retain_graph=True)

    def run_invocations_on_batch(
        self,
        batch: Batch,
        invocations: Invocations,
        order: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
        target: Optional[List[Batch]] = None,
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
                activations[invocation.order][batch.index] = self.run_invocation(
                    batch, partition, skip_trackers, invocation, target=target
                )
            elif invocation.source and invocation.source.stage == self.group.rank():
                invocations_handled += 1
                last_order = invocation.order
                batch = activations[invocation.order - 1][batch.index]
                activations[invocation.order][batch.index] = self.run_invocation(
                    batch, partition, skip_trackers, invocation, target=target
                )
                del activations[invocation.order - 1][batch.index]

            elif invocation.source and invocation.source.stage != self.group.rank():
                break

        return (invocations_handled, last_order)

    def event_loop_head(
        self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals]
    ) -> Optional[BackwardContext]:
        """The event loop for the "head", which first performs the forward pass
        on any applicable layers for this stage, and then enters the common
        `event_loop_inner`"""

        invocations, activations = self.get_invocations_and_activations()

        count_per_order: Dict[int, int] = dict()
        num_batches = len(batches)

        self.event_loop_inner(
            num_batches,
            skip_trackers,
            activations,
            invocations,
            count_per_order,
            ignore_gradients=True,
            batches=batches,
        )

        if self.training:
            return BackwardContext(activations, invocations, count_per_order, num_batches=num_batches)
        else:
            return None

    def head_backwards(self, head_ctx: BackwardContext) -> None:
        assert head_ctx.num_batches is not None
        assert self.training
        self.event_loop_inner(
            head_ctx.num_batches,
            [],
            head_ctx.activations,
            head_ctx.invocations,
            head_ctx.count_per_order,
            ignore_activations=True,
        )

    def get_batch_from_message(self, message: PipeMessage) -> Batch:
        """Get the tensor(s) wrapped in a `Batch` from a `PipeMessage`, applying
        AsyncRecvOperator so we can intercept the backward pass"""

        microbatch_index = message.args.microbatch_index
        phony = torch.empty(0, device=self.transport.input_device, requires_grad=True)
        result = AsyncRecvOperator.apply(phony, self.transport, message)
        if len(result) == 1:
            batch = Batch(result[0], microbatch_index)
        else:
            batch = Batch(result, microbatch_index)
        return batch

    def event_loop_tail(
        self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals], target: Optional[List[Batch]] = None,
    ) -> None:
        """The event loop for the "tail", or final stage which only processes
        activations and then returns to the caller so that the loss can be
        calculated. This also handles the first/only stage for the special
        case of a 1-stage pipeline."""

        assert self.group

        invocations, activations = self.get_invocations_and_activations()

        rank = self.group.rank()

        self.event_loop_inner(
            len(batches),
            skip_trackers,
            activations,
            invocations,
            count_per_order=dict(),
            ignore_gradients=True,
            tail=True,
            batches=batches if rank == 0 else None,
            target=target,
        )

        _, last_invocation = invocations.popitem()

        for index, batch in activations[last_invocation.order].items():
            batches[index] = batch

    def get_invocations_and_activations(self) -> Tuple[Invocations, Activations]:
        activations: Activations = dict()
        invocations: Invocations = OrderedDict()

        for partition in self.partitions:
            for invocation in partition.invocations:
                activations[invocation.order] = dict()
                invocations[invocation.order] = invocation

        invocations = OrderedDict(sorted(invocations.items(), key=lambda entry: entry[0]))

        return (invocations, activations)

    def event_loop(self, num_microbatch: int, skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "middle", i.e. neither the head nor the tail"""
        assert self.group

        invocations, activations = self.get_invocations_and_activations()

        self.event_loop_inner(num_microbatch, skip_trackers, activations, invocations, dict())

    def event_loop_inner(
        self,
        num_batches: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
        invocations: Invocations,
        count_per_order: Dict[int, int],
        *,
        batches: Optional[List[Batch]] = None,
        ignore_activations: bool = False,
        ignore_gradients: bool = False,
        tail: bool = False,
        target: Optional[List[Batch]] = None,
    ) -> None:
        """The common event loop shared by all stages. This processses
        activations for the forward pass, and if `self.training` is true,
        processes gradients for the backward pass."""

        expected_invocations = num_batches * len(invocations)

        if self.training and not ignore_gradients:
            num_gradients = 0
        else:
            num_gradients = expected_invocations

        if ignore_activations:
            num_activations = expected_invocations
        else:
            num_activations = 0

        stashed_messages: Dict[Tuple[int, int], PipeMessage] = dict()
        batch_iter = batches or []
        if tail:
            # Make a copy of the list as event_loop_tail modifies batches after
            batch_iter = list(batch_iter)

        receiving_invocations = sum(inv.receives_activation() for inv in invocations.values())
        expected_recv = receiving_invocations * num_batches

        mp_group = get_model_parallel_group()

        irecvr: Optional[IRecvWrapper] = None
        if mp_group.rank() == 0 and receiving_invocations > 0 and num_activations < expected_invocations:
            irecvr = self.transport.get_irecv_wrapper(self.group, EVENT_LOOP_QUEUE)
            if irecvr:
                irecvr.maybe_irecv()

        sequence_group = get_model_parallel_group_with_pipeline_backend()

        def maybe_get_message() -> Optional[PipeMessage]:
            if irecvr and irecvr.is_active() and (irecvr.is_completed() or len(batch_iter) == 0):
                message_tensor = irecvr.wait()
                return self.transport.recv_message_header(EVENT_LOOP_QUEUE, tensor=message_tensor)
            elif not batch_iter:
                return self.transport.recv_message_header(EVENT_LOOP_QUEUE)
            return None

        def get_message_with_sequence(expected_sequence: Tuple[int, int]) -> PipeMessage:
            if expected_sequence in stashed_messages:
                return stashed_messages.pop(expected_sequence)

            while True:
                message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
                sequence = (message.args.order, message.args.microbatch_index)

                if sequence == expected_sequence:
                    return message

                stashed_messages[sequence] = message

        next_gradient = dict()

        for inv in invocations.values():
            next_gradient[inv.order] = num_batches - 1

        while num_activations < expected_invocations or num_gradients < expected_invocations:
            message: Optional[PipeMessage] = None
            batch: Optional[Batch] = None

            if num_activations == expected_invocations and num_gradients < expected_invocations:
                for order in next_gradient:
                    value = next_gradient[order]
                    next_gradient[order] = -1
                    # print(f"maybe split {torch.distributed.get_rank()}, {value}")
                    if value >= 0:
                        split_batch = activations[order].get(value)
                        if split_batch is not None and split_batch.split is not None:
                            split_batch.split.backward()

            if mp_group.rank() == 0:
                message = maybe_get_message()
                if message:
                    sequence = (message.args.order, message.args.microbatch_index)
                else:
                    batch = batch_iter.pop(0)
                    sequence = (0, batch.index)

                if mp_group.size() > 1:
                    self.transport.send_sequence(sequence, sequence_group)
            else:
                expected_sequence = self.transport.recv_sequence(sequence_group)
                if batch_iter and expected_sequence[0] == 0:
                    batch = batch_iter.pop(0)
                else:
                    message = get_message_with_sequence(expected_sequence)

            if (message and message.args.message_type is AsyncMessageType.Activations) or batch:
                if message:
                    batch = self.get_batch_from_message(message)
                    invocation = invocations[message.args.order]
                    expected_recv -= 1
                else:
                    assert batch
                    invocation = invocations[0]

                inv_count, last_order = self.run_invocations_on_batch(
                    batch, invocations, invocation.order, skip_trackers, activations, target=target,
                )

                count_per_order[last_order] = inv_count
                num_activations += inv_count
                if tail and invocations[last_order].dest is None:
                    self.prepare_tail_backward(batch, activations, invocations, count_per_order)

                del batch

                assert num_activations <= expected_invocations

                if irecvr and expected_recv > 0:
                    irecvr.maybe_irecv()

            elif message and message.args.message_type is AsyncMessageType.Gradients:
                invocation = invocations[message.args.order]

                num_gradients += count_per_order[invocation.order]

                next_gradient[invocation.order] = message.args.microbatch_index - 1

                self.perform_backward_for_invocation(self.transport, message, activations, invocation)
                if irecvr and num_gradients < expected_invocations:
                    irecvr.maybe_irecv()
            else:
                raise ValueError("Missing message/bad type: {message}")

    @staticmethod
    def prepare_tail_backward(
        batch: Batch, activations: Activations, invocations: Invocations, count_per_order: Dict[int, int],
    ) -> None:
        """Called once per microbatch to handle the case where the final stage
        receives gradients from an earlier stage due to layer reuse"""

        expected_gradients = sum(inv.sends_activation() for inv in invocations.values())

        if expected_gradients > 0:
            grad_fn = next(b.grad_fn for b in batch if b.requires_grad)
            assert grad_fn
            grad_fn.tail_ctx = BackwardContext(
                activations, invocations, count_per_order, expected_recv=expected_gradients
            )
