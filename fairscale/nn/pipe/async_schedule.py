from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from fairscale.nn.model_parallel import get_pipeline_parallel_group, get_pipeline_parallel_ranks

from .messages import MESSAGE_TENSOR_SIZE, MessageQueues, send_message, tensor_to_pyobject, to_input_device
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals
from .types import EVENT_LOOP_QUEUE, InputDevice, PipelineStyle, PipeMessage, Tensors, TransportConfig

Activations = Dict[int, Dict[int, Dict[int, Batch]]]


def dprint(x: str) -> None:
    pass


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


class Hackity(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, *x):
        return torch.tensor(1.0)

    @staticmethod
    # type: ignore
    def backward(ctx, grad):
        assert ctx.grad_from_pipeline is not None
        return ctx.grad_from_pipeline


def recv_async_tensors(
    rank: int, input_device: InputDevice, config: TransportConfig, message: PipeMessage
) -> PipeMessage:
    if config.use_rpc:
        # Tensors already contained within message
        message.tensors = to_input_device(message.tensors, input_device)
        dprint(f"recv_async_tensors {torch.distributed.get_rank()}, {len(message.tensors)}")
        return message
    else:
        torch.cuda.current_stream().synchronize()

        message_tensors = []
        for index, (shape, dtype) in enumerate(zip(message.tensor_shapes, message.tensor_dtypes)):
            t = torch.empty(*shape, dtype=dtype, device=input_device)
            torch.distributed.recv(t, message.src, tag=message.tag + index, group=get_pipeline_parallel_group())
            message_tensors.append(t)

        message.tensors = tuple(message_tensors)

        torch.cuda.current_stream().synchronize()
        return message


class AsyncRecvOperator(torch.autograd.Function):
    """Receive activations to the previous pipeline stage"""

    @staticmethod
    # type: ignore
    def forward(
        ctx, dst_rank: int, phony: Tensor, input_device, config: TransportConfig, message: PipeMessage
    ) -> Tensors:
        assert dst_rank == torch.distributed.get_rank()
        ctx.config = config
        ctx.index = message.args.microbatch_index

        result = recv_async_tensors(dst_rank, input_device, config, message)

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
        dprint(f"AsyncRecvOperator back {this_rank} {len(grad)}, {ctx.args}")
        # Note that dst/source are swaped coz in backward pass, maybe abstract
        # this out?
        body = AsyncMessageBody(
            AsyncMessageType.Gradients, ctx.index, source=ctx.args.dest, dest=ctx.args.source, order=ctx.args.order - 1
        )
        dprint(f"AsyncRecvOperator 2 back {this_rank} {len(grad)}")
        send_message(
            ctx.config,
            PipeMessage(
                this_rank, ranks[ctx.args.source.stage], queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple(grad),
            ),
            sync=True,
        )
        dprint(f"AsyncRecvOperator 3 back {this_rank} {len(grad)}")
        return (None, None, None, None, None)


def recv_async_header(transport_config: TransportConfig, input_device: InputDevice) -> PipeMessage:
    if transport_config.use_rpc:
        queue = MessageQueues[EVENT_LOOP_QUEUE]
        result = queue.get()
        result.tensors = to_input_device(result.tensors, input_device)
        return result
    else:
        dprint(f"cactus")
        tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=input_device)
        torch.cuda.current_stream().synchronize()
        torch.distributed.recv(tensor, src=None, tag=EVENT_LOOP_QUEUE, group=get_pipeline_parallel_group())
        torch.cuda.current_stream().synchronize()
        dprint(f"cactus2")
        return tensor_to_pyobject(tensor.cpu())


class AsyncEventLoop:
    def __init__(
        self,
        partitions: List[ModuleWrapper],
        group: ProcessGroup,
        transport_config: TransportConfig,
        training: bool,
        input_device: InputDevice,
        checkpoint_stop: int,
    ):
        self.training = training
        self.input_device = input_device
        self.checkpoint_stop = checkpoint_stop
        self.transport_config = transport_config
        self.group = group
        self.partitions: List[ModuleWrapper] = partitions

    def send_async_message(
        self, src_rank: int, dst_rank: int, input: List[Tensor], index: int, invocation: Invocation
    ) -> None:
        assert src_rank == torch.distributed.get_rank()
        assert invocation.dest

        body = AsyncMessageBody(
            AsyncMessageType.Activations, index, invocation.this, invocation.dest, invocation.order + 1
        )
        dprint(f">>> send batch {src_rank} {dst_rank} {len(input)} {invocation.order}")
        send_message(
            self.transport_config,
            PipeMessage(src_rank, dst_rank, queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple(input)),
            sync=True,
        )
        dprint(f"<<< send batch {src_rank} {dst_rank} {len(input)} {invocation.order}")

    def async_send_inner(
        self,
        batch: Batch,
        partition: ModuleWrapper,
        index: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        invocation: Invocation,
    ) -> Batch:
        assert self.group
        from .pipeline import create_task

        task = create_task(
            PipelineStyle.AsyncSchedule,
            self.checkpoint_stop,
            index,
            self.group.rank(),
            batch,
            partition.module,
            skip_trackers,
            [],
        )
        result = task.compute()
        task.finalize(result)

        if invocation.dest and invocation.dest.stage != invocation.this.stage:
            ranks = get_pipeline_parallel_ranks()
            this_rank = torch.distributed.get_rank()

            # self.send_skip_tensors(this_rank, ranks, batch, i, skip_trackers)
            dprint(f"sending to next stage from {this_rank}...{invocation}, {index}")
            self.send_async_message(this_rank, ranks[invocation.dest.stage], [*result], index, invocation)
            z = Hackity.apply(*result)
            result = Batch(z, result.index)
            dprint(f"empty yay!")
        else:
            dprint(f"not sending to next stage...{invocation.this}, {invocation.dest}")
        return result

    def async_grad_inner(self, message: PipeMessage, activations: Activations, invocation: Invocation) -> None:
        args: AsyncMessageBody = message.args
        if self.transport_config.use_rpc:
            recvd_grads = message
        else:
            recvd_grads = recv_async_tensors(
                torch.distributed.get_rank(), self.input_device, self.transport_config, message
            )

        # FIXME tom

        batch: Batch = activations[invocation.this.index][invocation.order][args.microbatch_index]

        try:
            batch.tensor.grad_fn.grad_from_pipeline = tuple(recvd_grads.tensors)  # type: ignore
            batch.tensor.backward(retain_graph=True)
            return
        except Exception as e:
            print(f"hackity fail {e}")
            raise e

    def process_batch_forward(
        self,
        batch: Batch,
        i: int,
        invocations: List[Invocation],
        order: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
    ) -> Tuple[int, int]:
        invocations_handled = 0
        last_order = 0
        for invocation in invocations:
            if invocation.order < order:
                continue
            pi = invocation.this.index
            partition = self.partitions[pi]

            dprint(f"{self.group.rank()}: pbb {invocation}, {order}, {self.group.rank()}")
            if invocation.order == order:
                dprint(f"{self.group.rank()}: assigning {pi}, {invocation.order}, {i}")
                invocations_handled += 1
                last_order = invocation.order
                activations[pi][invocation.order][i] = self.async_send_inner(
                    batch, partition, i, skip_trackers, invocation
                )
            elif invocation.source and invocation.source.stage == self.group.rank():
                dprint(
                    f"{self.group.rank()}: reading {invocation}, {invocation.source.index}, {invocation.order-1}, {i}"
                )
                invocations_handled += 1
                last_order = invocation.order
                batch = activations[invocation.source.index][invocation.order - 1][i]
                dprint(f"{self.group.rank()}: assigning {pi}, {invocation.order}, {i}")
                activations[pi][invocation.order][i] = self.async_send_inner(
                    batch, partition, i, skip_trackers, invocation
                )
                del activations[invocation.source.index][invocation.order - 1][i]

            elif invocation.source and invocation.source.stage != self.group.rank():
                break

        dprint(f"pbb {self.group.rank()} {invocations_handled}")
        return (invocations_handled, last_order)

    def event_loop_head(
        self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals], event: Optional[Event]
    ) -> None:

        invocations, activations = self.get_sorted_invocations_and_activations()

        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0

        count_per_order = dict()

        dprint(f"head loop start {torch.distributed.get_rank()}")
        for i, batch in enumerate(batches):
            dprint(f"head loop iter {torch.distributed.get_rank()}, {i}")
            inv_count, last_order = self.process_batch_forward(batch, i, invocations, 0, skip_trackers, activations)
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count

        dprint(f"head wat {actual_invocations}, {expected_invocations}")
        if actual_invocations < expected_invocations or self.training:
            dprint(f"head extra {actual_invocations}, {expected_invocations}")
            self.event_loop_inner(
                expected_invocations,
                skip_trackers,
                activations,
                invocations,
                count_per_order,
                already_received=actual_invocations,
                event=event,
            )

        # if self.pipeline.training:
        #    for _ in range(len(batches)):
        #        message = self.recv_async_header()
        #        args: AsyncMessageBody = message.args
        #        assert args.message_type is AsyncMessageType.Gradients
        #        self.async_grad_inner(message, activations)

    def event_loop_tail(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        assert self.group

        invocations, activations = self.get_sorted_invocations_and_activations()
        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0

        rank = self.group.rank()
        count_per_order = dict()

        for i, batch in enumerate(batches):
            if rank == 0:
                batch_index = i
                order = 0
            else:
                message = recv_async_header(self.transport_config, self.input_device)
                args: AsyncMessageBody = message.args

                phony = torch.empty(0, device=self.input_device, requires_grad=True)
                result = AsyncRecvOperator.apply(
                    torch.distributed.get_rank(), phony, self.input_device, self.transport_config, message,
                )
                if len(result) == 1:
                    batch = Batch(result[0], args.microbatch_index)
                else:
                    batch = Batch(result, args.microbatch_index)
                batch_index = args.microbatch_index
                order = args.order

            inv_count, last_order = self.process_batch_forward(
                batch, batch_index, invocations, order, skip_trackers, activations
            )
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count

        if actual_invocations < expected_invocations:
            expected_gradients = 0  # (len(invocations) - 1) * len(batches)
            dprint(f"tail expect {expected_invocations}, {len(invocations)}, {len(batches)}")

            self.event_loop_inner(
                expected_invocations,
                skip_trackers,
                activations,
                invocations,
                count_per_order,
                already_received=actual_invocations,
                ignore_gradients=True,
            )

        for index, batch in activations[len(self.partitions) - 1][invocations[-1].order].items():
            batches[index] = batch

    def get_sorted_invocations_and_activations(self) -> Tuple[List[Invocation], Activations]:
        activations: Activations = dict()
        invocations: List[Invocation] = []

        for pi, partition in enumerate(self.partitions):
            activations[pi] = dict()
            for invocation in partition.invocations:
                activations[pi][invocation.order] = dict()
                invocations.append(invocation)

        invocations.sort(key=lambda inv: inv.order)

        return (invocations, activations)

    def event_loop(self, num_microbatch: int, skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        assert self.group

        invocations, activations = self.get_sorted_invocations_and_activations()

        expected_invocations = len(invocations) * num_microbatch

        dprint(f"event_loop {expected_invocations}, {num_microbatch}, {len(invocations)}")
        self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, dict())

    def event_loop_inner(
        self,
        expected_invocations: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
        invocations: List[Invocation],
        count_per_order: Dict[int, int],
        *,
        already_received: int = 0,
        ignore_gradients: bool = False,
        event: Optional[Event] = None,
    ) -> None:

        num_activations = already_received
        if self.training and not ignore_gradients:
            num_gradients = 0
        else:
            num_gradients = expected_invocations

        while num_activations < expected_invocations or num_gradients < expected_invocations:
            dprint(
                f">> recv_async_header {self.group.rank()}, {torch.distributed.get_rank()} {expected_invocations},"
                f" {num_activations}, {num_gradients}, {ignore_gradients}"
            )
            if num_activations == expected_invocations and num_gradients == 0 and event is not None:
                print(f">>> wait on event")
                event.wait()
                print(f"<<< wait on event")

            message = recv_async_header(self.transport_config, self.input_device)
            dprint(f"<< recv_async_header {torch.distributed.get_rank()}")
            args: AsyncMessageBody = message.args

            filtered = [inv for inv in invocations if inv.order == args.order]
            if len(filtered) == 0:
                dprint(f"no invocation on {self.group.rank()} for {args.order}, {invocations}")
            invocation = filtered[0]

            # FIXME(tom) for combining pipeline with megatron, I currently don't
            # control the order of received activations or gradients, so it is
            # possible for a reused ColumnParallelLinear for example to receive
            # a different order of activations w.r.t. the sending stage, which
            # would result in incorrect values being used for the all_gather
            if args.message_type is AsyncMessageType.Activations:

                phony = torch.empty(0, device=self.input_device, requires_grad=True)
                result = AsyncRecvOperator.apply(
                    torch.distributed.get_rank(), phony, self.input_device, self.transport_config, message,
                )

                dprint(
                    f"got batch {torch.distributed.get_rank()}|{self.group.rank()} i:{args.microbatch_index}"
                    f" len:{len(result)}, {invocation}"
                )

                if len(result) == 1:
                    batch = Batch(result[0], args.microbatch_index)
                else:
                    batch = Batch(result, args.microbatch_index)

                dprint(f"calling pbb? {self.group.rank()}, {expected_invocations}, {num_activations}, {num_gradients}")
                inv_count, last_order = self.process_batch_forward(
                    batch, args.microbatch_index, invocations, args.order, skip_trackers, activations
                )
                count_per_order[last_order] = inv_count
                num_activations += inv_count
                assert num_activations <= expected_invocations

            elif args.message_type is AsyncMessageType.Gradients:
                dprint(f">> try {self.group.rank()}, {invocation.order}, {count_per_order}, {num_gradients}")
                num_gradients += count_per_order[invocation.order]
                self.async_grad_inner(message, activations, invocation)
                dprint(f"<< try {self.group.rank()}, {invocation.order}, {count_per_order}, {num_gradients}")
