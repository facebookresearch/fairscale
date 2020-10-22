# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank

from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group

from . import Pipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

PipeModel: Pipe
PipeResult: TensorOrTensors


SizeOrSizes = Union[torch.Size, List[torch.Size]]
DtypeOrDtypes = Union[torch.dtype, List[torch.dtype]]


def set_device_based_on_group(group: ProcessGroup) -> None:
    # torch.cuda.set_device(group.rank() % torch.cuda.device_count())
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())


def register_remote_model(args: List[Any], kwargs: Dict[str, Any]) -> None:
    group = get_pipeline_parallel_group()  # FIXME(tom) handle dynamic group
    set_device_based_on_group(group)
    kwargs["group"] = group
    kwargs["input_device"] = torch.device("cuda", torch.cuda.current_device())
    model = Pipe(*args, **kwargs)
    model.cuda()
    global PipeModel
    PipeModel = model


def get_shapes(tensor: TensorOrTensors) -> SizeOrSizes:
    if isinstance(tensor, torch.Tensor):
        return tensor.shape
    else:
        return [t.shape for t in tensor]


def get_dtype(tensor: TensorOrTensors) -> DtypeOrDtypes:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    else:
        return [t.dtype for t in tensor]


def model_forward(training: bool, shape: torch.Size, dtype: torch.dtype) -> Optional[Tuple[SizeOrSizes, DtypeOrDtypes]]:
    if isinstance(shape, torch.Size):
        tensor = torch.empty(shape, dtype=dtype)
    else:
        tensor = tuple([torch.empty(s, dtype=d) for s, d in zip(shape, dtype)])

    model = PipeModel
    assert model.group
    set_device_based_on_group(model.group)

    model.train(training)
    result = model(tensor)
    if model.final_stage:
        global PipeResult
        PipeResult = result
        return (get_shapes(result), get_dtype(result))

    return None


def send_result_and_do_backwards(training: bool, message: PipeMessage, grads_message: PipeMessage) -> None:
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)
    result = PipeResult
    model = PipeModel

    if isinstance(result, torch.Tensor):
        result = tuple([result])

    message.tensors = tuple(result)
    assert model.pipeline
    transport = model.pipeline.transport
    transport.send_message(message, sync=False, skip_header=True)

    if training:
        grads_message.tensor_shapes = [r.shape for r in result]
        grads_message.tensor_dtypes = [r.dtype for r in result]
        grads_message = transport.recv_message_tensors(grads_message)

        with model.lock:
            torch.autograd.backward(result, grads_message.tensors, retain_graph=True)


def recv_result(model: Pipe, shapes: SizeOrSizes, dtypes: DtypeOrDtypes, message: PipeMessage) -> TensorOrTensors:
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)

    assert model.pipeline
    transport = model.pipeline.transport

    if isinstance(shapes, torch.Size):
        message.tensor_shapes = [cast(torch.Size, shapes)]
        message.tensor_dtypes = [cast(torch.dtype, dtypes)]
        message = transport.recv_message_tensors(message)
        return message.tensors[0]
    else:
        message.tensor_shapes = cast(List[torch.Size], shapes)
        message.tensor_dtypes = cast(List[torch.dtype], dtypes)
        message = transport.recv_message_tensors(message)
        return message.tensors


def get_global_ranks_from_group(group: ProcessGroup) -> List[int]:
    return [_get_global_rank(group, r) for r in range(group.size())]


def model_forward_first_stage(model: Pipe, tensor: TensorOrTensors, event: Event, lock: Lock) -> None:
    t = model.training
    with lock:
        assert model.group
        set_device_based_on_group(model.group)
        model(tensor, event=event)


class PipeBackRedirect(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, inputs, dest, event, message, transport):
        ctx.dest = dest
        ctx.event = event
        ctx.message = message
        ctx.transport = transport
        return inputs

    @staticmethod
    # type: ignore
    def backward(ctx, *grad):
        ctx.message.tensors = tuple(grad)
        ctx.transport.send_message(ctx.message, sync=False, skip_header=True)
        ctx.event.set()
        return (None, None, None, None, None)


def callback_with_model(callback: Callable, ctx: Any) -> None:
    group = get_pipeline_parallel_group()  # FIXME(tom) handle dynamic group
    set_device_based_on_group(group)

    with PipeModel.lock:
        callback(ctx, PipeModel)


class PipeRPCWrapper(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.group = cast(ProcessGroup, kwargs.get("group")) or get_pipeline_parallel_group()
        assert self.group.rank() == 0
        self.lock = Lock()

        if True:
            assert (
                self.group == get_pipeline_parallel_group()
            ), "Can't pickle groups, so group must be `get_pipeline_parallel_group()`"
            kwargs["group"] = None
        else:
            kwargs["group"] = self.group

        kwargs["style"] = Pipe.AsyncSchedule
        kwargs["input_device"] = torch.device("cuda", torch.cuda.current_device())

        self.model = Pipe(*args, **kwargs)
        self.worker_map = kwargs["worker_map"]
        futures = [
            # FIXME get global rank
            rpc.rpc_async(self.get_rpc_name(rank), register_remote_model, args=(args, kwargs))
            for rank in range(1, self.group.size())
        ]
        futures = [f.wait() for f in futures]
        self.model.cuda()

    def get_rpc_name(self, rank: int) -> str:
        return self.worker_map[_get_global_rank(self.group, rank)]

    def foreach_worker(self, callback: Callable, ctx: Any = None, *, include_self: bool = False) -> None:
        futures = [
            rpc.rpc_async(self.get_rpc_name(rank), callback_with_model, args=(callback, ctx))
            for rank in range(1, self.group.size())
        ]
        futures = [f.wait() for f in futures]
        if include_self:
            with self.model.lock:
                callback(ctx, self.model)

    def forward(self, tensor: TensorOrTensors) -> TensorOrTensors:  # type: ignore
        shape = get_shapes(tensor)
        dtype = get_dtype(tensor)

        if isinstance(tensor, torch.Tensor):
            num_tensors = 1
        else:
            num_tensors = len(tensor)

        futures = [
            rpc.rpc_async(self.get_rpc_name(rank), model_forward, args=(self.model.training, shape, dtype))
            for rank in range(1, self.group.size())
        ]

        if self.model.final_stage:
            return self.model(tensor)
        else:
            event = Event()
            t = Thread(target=model_forward_first_stage, args=(self.model, tensor, event, self.lock))
            t.start()

            shape, dtype = futures[-1].wait()
            dest_rank = self.group.size() - 1
            dest = self.get_rpc_name(dest_rank)
            dest_global_rank = _get_global_rank(self.group, dest_rank)
            src_global_rank = torch.distributed.get_rank()
            queue = EVENT_LOOP_QUEUE

            activations = PipeMessage(dest_global_rank, src_global_rank, queue_name=queue, tensor_count=num_tensors)
            grads = PipeMessage(src_global_rank, dest_global_rank, queue_name=queue, tensor_count=num_tensors)

            rpc.rpc_async(dest, send_result_and_do_backwards, args=(self.model.training, activations, grads))
            result = recv_result(self.model, shape, dtype, activations)
            if isinstance(result, torch.Tensor):
                result.requires_grad_()
            else:
                for r in result:
                    r.requires_grad_()

            assert self.model.pipeline
            return PipeBackRedirect.apply(result, dest_global_rank, event, grads, self.model.pipeline.transport)

    @property
    def final_stage(self) -> bool:
        return self.model.final_stage
