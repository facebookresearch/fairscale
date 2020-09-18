import os
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank

from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group

from . import Pipe
from .types import TensorOrTensors

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

PipeModel: Pipe


SizeOrSizes = Union[torch.Size, List[torch.Size]]
DtypeOrDtypes = Union[torch.dtype, List[torch.dtype]]


def dprint(s: str) -> None:
    print(str(torch.distributed.get_rank()) + ": " + s)


def set_device_based_on_group(group: ProcessGroup) -> None:
    # torch.cuda.set_device(group.rank() % torch.cuda.device_count())
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())


def register_remote_model(args: List[Any], kwargs: Dict[str, Any]) -> None:
    group = get_pipeline_parallel_group()  # FIXME(tom) handle dynamic group
    set_device_based_on_group(group)
    dprint(f"model registered {torch.cuda.current_device()}")
    kwargs["group"] = group
    kwargs["input_device"] = torch.device("cuda", torch.cuda.current_device())
    model = Pipe(*args, **kwargs)
    model.cuda()
    globals()["PipeModel"] = model


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
    try:
        dprint(f"mf: train stage {torch.distributed.get_rank()}")
        if isinstance(shape, torch.Size):
            tensor = torch.empty(shape, dtype=dtype)
        else:
            tensor = tuple([torch.empty(s, dtype=d) for s, d in zip(shape, dtype)])

        model = globals()["PipeModel"]
        set_device_based_on_group(model.group)

        dprint(f"mf: train stage {model.group.rank()}, {os.getpid()}")
        model.train(training)
        result = model(tensor)
        torch.cuda.current_stream().synchronize()
        if model.final_stage:
            globals()["PipeResult"] = result
            return (get_shapes(result), get_dtype(result))
    except Exception as e:
        print(f"failboat {e} {type(e)}")
        import traceback

        print(f"format {traceback.format_exc()}")
        raise e

    return None


def send_result(training: bool) -> None:
    dprint(f"send result {training}")
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)
    try:
        dprint(f"send result {torch.distributed.get_rank()}, {torch.cuda.current_device()}")
        result = globals()["PipeResult"]
        model = globals()["PipeModel"]

        if isinstance(result, torch.Tensor):
            result = [result]

        dest = _get_global_rank(group, 0)
        print(
            f"ho har {torch.distributed.get_rank()} " + str([_get_global_rank(group, i) for i in range(group.size())])
        )
        torch.cuda.current_stream().synchronize()
        for r in result:
            dprint(f">>> send {torch.distributed.get_rank()}, {dest} {r.shape}, {r.dtype}, {r.device}")
            if "Gloo" in group.__class__.__name__:
                r = r.cpu()
            torch.distributed.send(r.contiguous(), dest, group=group)
            dprint(f"<<< send {torch.distributed.get_rank()}, {dest}")
        torch.cuda.current_stream().synchronize()

        if training:
            grads = []
            for r in result:
                g = torch.empty(r.shape).cuda()
                dprint(f">>> recv grads {g.shape}")
                torch.cuda.current_stream().synchronize()
                torch.distributed.recv(g, dest, group=group)
                torch.cuda.current_stream().synchronize()
                if "Gloo" in group.__class__.__name__:
                    g = g.cuda()
                dprint(f"<<< recv grads {g.shape}")
                grads.append(g)

            with model.lock:
                print(f" >>> autograd-backward tail")
                torch.autograd.backward(result, tuple(grads), retain_graph=True)
                print(f" <<< autograd-backward tail")
                torch.cuda.synchronize()

    except Exception as e:
        print(f"got {e}")


def recv_result(shapes: SizeOrSizes, dtypes: DtypeOrDtypes) -> TensorOrTensors:
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)
    src = torch.distributed.distributed_c10d._get_global_rank(group, group.size() - 1)
    dprint(f"recv_result... {src}, {torch.cuda.current_device()}")

    if isinstance(shapes, torch.Size):
        shape = cast(torch.Size, shapes)
        dtype = cast(torch.dtype, dtypes)
        t = torch.empty(shape, dtype=dtype).cuda()
        dprint(f">>> recv {torch.distributed.get_rank()}, {src} {t.shape}, {t.dtype}")
        torch.cuda.current_stream().synchronize()
        torch.distributed.recv(t, src, group=group)
        torch.cuda.current_stream().synchronize()
        if "Gloo" in group.__class__.__name__:
            t = t.cuda()
        dprint(f"<<< recv {torch.distributed.get_rank()}, {src}")
        dprint(f"recvd solo")
        return t
    else:
        result = []
        torch.cuda.current_stream().synchronize()
        shapes = cast(List[torch.Size], shapes)
        dtypes = cast(List[torch.dtype], dtypes)
        for s, d in zip(shapes, dtypes):
            t = torch.empty(s, dtype=d).cuda()
            dprint(f">>> recv {torch.distributed.get_rank()}, {src} {t.shape}, {t.dtype}")
            torch.distributed.recv(t, src, group=group)
            if "Gloo" in group.__class__.__name__:
                t = t.cuda()
            dprint(f"<<< recv {torch.distributed.get_rank()}, {src}")
            dprint(f"recvd multi / {len(shapes)}")
            result.append(t)
        torch.cuda.current_stream().synchronize()
        return tuple(result)


def get_global_ranks_from_group(group: ProcessGroup) -> List[int]:
    return [torch.distributed.distributed_c10d._get_global_rank(group, r) for r in range(group.size())]


def run_model(model: Pipe, tensor: TensorOrTensors, event: Event, lock: Lock) -> None:
    t = model.training
    with lock:
        print(f">> run_model thread {t}")
        assert model.group
        set_device_based_on_group(model.group)
        torch.cuda.current_stream().synchronize()
        torch.cuda.synchronize()
        model(tensor, event=event)
        torch.cuda.synchronize()
        torch.cuda.current_stream().synchronize()
        print(f"<< run_model thread {t}")


class PipeBackRedirect(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, inputs, dest, event):
        ctx.dest = dest
        ctx.event = event
        return inputs

    @staticmethod
    # type: ignore
    def backward(ctx, *grad):
        dprint(f">>> back hook yay")
        group = get_pipeline_parallel_group()
        torch.cuda.current_stream().synchronize()
        for g in grad:
            dprint(f">>> back send {g.shape}")
            if "Gloo" in group.__class__.__name__:
                g = g.cpu()
            torch.distributed.send(g, ctx.dest, group=group)
            dprint(f"<<< back send")
        torch.cuda.current_stream().synchronize()
        ctx.event.set()
        dprint(f"<<< back hook yay")
        return (None, None, None, None)


def callback_with_model(callback: Callable, ctx: Any) -> None:
    group = get_pipeline_parallel_group()  # FIXME(tom) handle dynamic group
    set_device_based_on_group(group)

    global PipeModel

    torch.cuda.current_stream().synchronize()
    with PipeModel.lock:
        callback(ctx, PipeModel)
    torch.cuda.current_stream().synchronize()


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
        print(f"calling rpc {args}, {kwargs}")
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

        futures = [
            rpc.rpc_async(self.get_rpc_name(rank), model_forward, args=(self.model.training, shape, dtype))
            for rank in range(1, self.group.size())
        ]

        if self.model.final_stage:
            return self.model(tensor)
        else:
            event = Event()
            t = Thread(target=run_model, args=(self.model, tensor, event, self.lock))
            t.start()

            dprint("forward before wait recv")
            shape, dtype = futures[-1].wait()
            dprint("forward after wait recv")
            dest_rank = self.group.size() - 1
            dest = self.get_rpc_name(dest_rank)
            dprint(f"async to {dest}")
            rpc.rpc_async(dest, send_result, args=(self.model.training,))
            dprint(">>> recv_result")
            result = recv_result(shape, dtype)
            dprint("<<< recv_result")
            # event.set()
            dprint("not set event")
            try:
                if isinstance(result, torch.Tensor):
                    result.requires_grad_()
                else:
                    for r in result:
                        r.requires_grad_()

                applied = PipeBackRedirect.apply(result, _get_global_rank(self.group, dest_rank), event)
            except Exception as e:
                dprint(f"failed got {e}")
            dprint("return applied")
            return applied

    @property
    def final_stage(self) -> bool:
        return self.model.final_stage
