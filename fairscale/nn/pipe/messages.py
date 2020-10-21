import pickle
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Any, List

import numpy as np
import torch
from torch import Tensor

from fairscale.nn.model_parallel import get_pipeline_parallel_group

from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors, TransportConfig

# FIXME Why is 256 ok for training but not for tests?
MESSAGE_TENSOR_SIZE = 1024  # 256

MessageQueues: List[Queue] = [Queue() for _ in range(MESSAGE_GENERATION_START)]


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
    try:
        nparray = tensor.numpy()
        return pickle.loads(nparray.tobytes())
    except Exception as e:
        print(f"pickle fail {e}")
        raise e


def to_input_device(tensors: Tensors, input_device: InputDevice) -> Tensors:
    if input_device is None:
        return tensors
    else:
        return tuple(t.to(input_device) for t in tensors)


def rpc_push_queue(message: PipeMessage) -> None:
    globals()["MessageQueues"][message.queue_name].put(message)


def send_message(config: TransportConfig, message: PipeMessage, sync: bool = False, skip_header: bool = False) -> None:
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
        torch.cuda.current_stream().synchronize()
        if not skip_header:
            message.tensor_shapes = [t.size() for t in tensors]
            message.tensor_dtypes = [t.dtype for t in tensors]
            torch.distributed.send(
                pyobject_to_tensor(message), message.dest, tag=message.queue_name, group=get_pipeline_parallel_group()
            )
        for index, t in enumerate(tensors):
            if t.device.type == "cpu":
                t = t.cuda()
            torch.distributed.send(
                t.contiguous(), message.dest, tag=message.tag + index, group=get_pipeline_parallel_group()
            )


def recv_message_header(transport_config: TransportConfig, input_device: InputDevice, queue_name: int) -> PipeMessage:
    if transport_config.use_rpc:
        queue = MessageQueues[queue_name]
        result = queue.get()
        result.tensors = to_input_device(result.tensors, input_device)
        return result
    else:
        tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=input_device)
        torch.cuda.current_stream().synchronize()
        torch.distributed.recv(tensor, src=None, tag=queue_name, group=get_pipeline_parallel_group())
        torch.cuda.current_stream().synchronize()
        return tensor_to_pyobject(tensor.cpu())


def recv_message_tensors(config: TransportConfig, message: PipeMessage) -> PipeMessage:
    if config.use_rpc:
        # Tensors already contained within message
        message.tensors = to_input_device(message.tensors, config.input_device)
        return message
    else:
        torch.cuda.current_stream().synchronize()

        message_tensors = []
        for index, (shape, dtype) in enumerate(zip(message.tensor_shapes, message.tensor_dtypes)):
            t = torch.empty(*shape, dtype=dtype, device=config.input_device)
            torch.distributed.recv(t, message.src, tag=message.tag + index, group=get_pipeline_parallel_group())
            message_tensors.append(t)

        message.tensors = tuple(message_tensors)

        torch.cuda.current_stream().synchronize()
        return message


def recv_message(
    config: TransportConfig, queue_name: int, *, nowait: bool = False, input_device: InputDevice = None
) -> PipeMessage:
    if config.use_rpc:
        queue = globals()["MessageQueues"][queue_name]
        if nowait:
            result = queue.get_nowait()
        else:
            result = queue.get()
        return recv_message_tensors(config, result)
    else:
        # FIXME(handle nowait)
        if nowait:
            raise QueueEmpty

        torch.cuda.current_stream().synchronize()
        tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=input_device)
        torch.distributed.recv(tensor, src=None, tag=queue_name, group=get_pipeline_parallel_group())
        torch.cuda.current_stream().synchronize()
        message = tensor_to_pyobject(tensor.cpu())

        return recv_message_tensors(config, message)


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
