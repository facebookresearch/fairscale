# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional, Tuple, cast

from dataclasses import dataclass
import torch
from torch.distributed.distributed_c10d import _get_global_rank

from fairscale.nn.model_parallel import get_pipeline_parallel_group
from fairscale.utils.object import pyobject_to_tensor, tensor_to_pyobject

from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors

MESSAGE_TENSOR_SIZE = 1024

MessageQueues: List[Queue] = [Queue() for _ in range(MESSAGE_GENERATION_START)]


def to_input_device(tensors: Tensors, input_device: InputDevice) -> Tensors:
    if input_device is None:
        return tensors
    else:
        return tuple(t.to(input_device) for t in tensors)


def rpc_push_queue(message: PipeMessage) -> None:
    globals()["MessageQueues"][message.queue_name].put(message)


@dataclass(frozen=True)
class Transport(ABC):
    worker_map: Optional[Dict[int, str]]
    input_device: InputDevice

    def recv_message(self, queue_name: int, *, nowait: bool = False) -> PipeMessage:
        message = self.recv_message_header(queue_name, nowait)
        return self.recv_message_tensors(message)

    def recv_message_header(
        self, queue_name: int, nowait: bool = False, tensor: Optional[torch.Tensor] = None
    ) -> PipeMessage:
        ...

    def recv_message_tensors(self, message: PipeMessage) -> PipeMessage:
        ...

    def send_message(self, message: PipeMessage, sync: bool = False, skip_header: bool = False) -> None:
        ...

    def get_out_of_order(self, queue_name: int, index: int) -> Tensors:
        ...

    def send_sequence(self, sequence: Tuple[int, int], group: torch.distributed.ProcessGroup) -> None:
        ...

    def recv_sequence(self, group: torch.distributed.ProcessGroup) -> Tuple[int, int]:
        ...

    def get_irecv_wrapper(self, group: torch.distributed.ProcessGroup, tag: int) -> Optional["IRecvWrapper"]:
        ...


def MakeTransport(use_rpc: bool, worker_map: Optional[Dict[int, str]], input_device: InputDevice) -> Transport:
    if use_rpc:
        if worker_map is None:
            raise ValueError("'RpcTransport' requires 'worker_map' to be set")
        return RpcTransport(worker_map, input_device)
    else:
        return SendRecvTransport(worker_map, input_device)


class IRecvWrapper:
    def __init__(self, input_device: InputDevice, group: torch.distributed.ProcessGroup, tag: int):
        self.tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=input_device)
        self.group = group
        self.tag = tag
        self.work_item: Optional[torch.distributed.Work] = None

    def maybe_irecv(self) -> None:
        if self.work_item is None:
            # FIXME(tom) should use irecv but src=None is not currently supported
            # see https://github.com/pytorch/pytorch/pull/47137
            if False:
                self.work_item = torch.distributed.irecv(self.tensor, src=None, tag=self.tag, group=self.group)
            else:
                self.work_item = self.group.recv_anysource([self.tensor], self.tag)

    def is_completed(self) -> bool:
        return (self.work_item is not None) and self.work_item.is_completed()

    def is_active(self) -> bool:
        return self.work_item is not None

    def wait(self) -> torch.Tensor:
        assert self.work_item
        self.work_item.wait()
        self.work_item = None
        return self.tensor


class RpcTransport(Transport):
    def send_message(self, message: PipeMessage, sync: bool = False, skip_header: bool = False) -> None:
        message.tensors = tuple(t.cpu() for t in message.tensors)
        assert self.worker_map
        name = self.worker_map[message.dest]
        if sync:
            torch.distributed.rpc.rpc_sync(name, rpc_push_queue, args=(message,))
        else:
            torch.distributed.rpc.rpc_async(name, rpc_push_queue, args=(message,))

    def recv_message_header(
        self, queue_name: int, nowait: bool = False, tensor: Optional[torch.Tensor] = None
    ) -> PipeMessage:
        assert tensor is None
        queue = MessageQueues[queue_name]
        if nowait:
            result = queue.get_nowait()
        else:
            result = queue.get()
        result.tensors = to_input_device(result.tensors, self.input_device)
        return result

    def recv_message_tensors(self, message: PipeMessage) -> PipeMessage:
        # Tensors already contained within message
        message.tensors = to_input_device(message.tensors, self.input_device)
        return message

    def get_out_of_order(self, queue_name: int, index: int) -> Tensors:
        """Receive a message with a known microbatch index, and handle out-of-order
        messages by placing them back on the queue"""

        queue = globals()["MessageQueues"][queue_name]
        out_of_order: List[PipeMessage] = []
        while True:
            message = self.recv_message(queue_name)
            got_index = message.args
            value = message.tensors
            if got_index == index:
                for b in out_of_order:
                    queue.put(b)
                return value
            else:
                out_of_order.append(message)

    def send_sequence(self, sequence: Tuple[int, int], group: torch.distributed.ProcessGroup) -> None:
        pass

    def recv_sequence(self, group: torch.distributed.ProcessGroup) -> Tuple[int, int]:
        return (0, 0)

    def get_irecv_wrapper(self, group: torch.distributed.ProcessGroup, tag: int) -> Optional[IRecvWrapper]:
        return None


class SendRecvTransport(Transport):
    def send_message(self, message: PipeMessage, sync: bool = False, skip_header: bool = False) -> None:
        torch.cuda.synchronize()
        tensors = message.tensors
        message.tensors = tuple()
        torch.cuda.current_stream().synchronize()
        if not skip_header:
            message.tensor_shapes = [t.size() for t in tensors]
            message.tensor_dtypes = [t.dtype for t in tensors]
            torch.distributed.send(
                pyobject_to_tensor(message, MESSAGE_TENSOR_SIZE).cuda(),
                message.dest,
                tag=message.queue_name,
                group=get_pipeline_parallel_group(),
            )
        for index, t in enumerate(tensors):
            if t.device.type == "cpu":
                t = t.cuda()
            tmp = t.contiguous()
            torch.cuda.current_stream().synchronize()
            torch.distributed.send(
                tmp.contiguous(), message.dest, tag=message.tag + index, group=get_pipeline_parallel_group()
            )

    def recv_message_header(
        self, queue_name: int, nowait: bool = False, tensor: Optional[torch.Tensor] = None
    ) -> PipeMessage:
        # FIXME(handle nowait)
        if nowait:
            raise QueueEmpty

        if tensor is None:
            tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=self.input_device)
            torch.cuda.current_stream().synchronize()
            sender = torch.distributed.recv(tensor, src=None, tag=queue_name, group=get_pipeline_parallel_group())
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        return tensor_to_pyobject(tensor)

    def recv_message_tensors(self, message: PipeMessage) -> PipeMessage:
        torch.cuda.current_stream().synchronize()

        message_tensors = []
        for index, (shape, dtype) in enumerate(zip(message.tensor_shapes, message.tensor_dtypes)):
            t = torch.empty(*shape, dtype=dtype, device=self.input_device)
            sender = torch.distributed.recv(
                t, message.src, tag=message.tag + index, group=get_pipeline_parallel_group()
            )
            message_tensors.append(t)

        message.tensors = tuple(message_tensors)

        torch.cuda.current_stream().synchronize()
        return message

    def get_out_of_order(self, queue_name: int, index: int) -> Tensors:
        message = self.recv_message(queue_name)
        assert message.args == index
        return message.tensors

    def send_sequence(self, sequence: Tuple[int, int], group: torch.distributed.ProcessGroup) -> None:
        assert group.rank() == 0
        tensor = torch.tensor(sequence, device=self.input_device)
        torch.cuda.current_stream().synchronize()
        for rank in range(1, group.size()):
            torch.distributed.send(tensor, _get_global_rank(group, rank), tag=1, group=group)

    def recv_sequence(self, group: torch.distributed.ProcessGroup) -> Tuple[int, int]:
        assert group.rank() != 0
        tensor = torch.tensor([0, 0], device=self.input_device)
        torch.distributed.recv(tensor, _get_global_rank(group, 0), tag=1, group=group)
        torch.cuda.current_stream().synchronize()
        return cast(Tuple[int, int], tuple(tensor.tolist()))

    def get_irecv_wrapper(self, group: torch.distributed.ProcessGroup, tag: int) -> Optional[IRecvWrapper]:
        return IRecvWrapper(self.input_device, group, tag)
