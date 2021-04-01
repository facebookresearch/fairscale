# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from threading import Condition
from types import TracebackType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union, cast

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import rpc

from fairscale.nn.pipe import microbatch
from fairscale.nn.pipe.checkpoint import Checkpointing
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.stream import (
    AbstractStream,
    current_stream,
    is_cuda,
    new_stream,
    use_device,
    use_stream,
    wait_stream,
)
from fairscale.nn.pipe.worker import Task, create_workers

Device = Union[torch.device, int, str]

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


if torch.__version__.split("+")[0].split(".")[:3] <= ["1", "8", "1"]:
    BOUNCE_TENSORS = True
else:
    BOUNCE_TENSORS = False


def rloss(loss_func: Callable, input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
    if BOUNCE_TENSORS:
        return loss_func(input_rref.remote().cpu().to_here(), target_rref.remote().cpu().to_here())
    else:
        return loss_func(input_rref.to_here(), target_rref.to_here())


def DistributedLoss(loss: nn.Module, *args: Tuple, **kwargs: Dict) -> Callable:
    loss_func = loss(*args, **kwargs)

    def dloss(input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
        return rpc.remote(input_rref.owner(), rloss, args=(loss_func, input_rref, target_rref))

    return dloss


class PipelineModule(nn.Module):
    def __init__(self, module_cls, args, kwargs={}, num_inputs=1, num_outputs=None, worker=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.module_args = (module_cls, args, kwargs)
        if worker is not None:
            self.instantiate(*worker)

    @staticmethod
    def _create_module(module_cls, args, kwargs, device):
        result = module_cls(*args, **kwargs)
        result.to(device)
        return result

    def instantiate(self, on, device):
        self.on = on
        self.device = device
        self.module_rref = rpc.remote(on, PipelineModule._create_module, self.module_args + (device,))
        return self

    def get_module_rref(self):
        return self.module_rref

    def forward(self, input):
        pass


class RemoteModuleSequence(nn.Module):
    def __init__(self, inputs=None):
        super().__init__()
        self.modules = []
        self.inputs = []

    def _add_new_module(self, num=1):
        for i in range(num):
            self.inputs.append(None)

    def _find_or_add(self, module):
        try:
            return self.modules.index(module)
        except ValueError:
            self._add_new_module()
            self.modules.append(module)
            return len(self.modules) - 1

    def add_sequence(self, modules, first_input=None):
        old_m = len(self.modules)
        m = len(modules)
        self._add_new_module(m)
        self.modules.extend(modules)
        for i in range(old_m + 1, old_m + m):
            self.inputs[i] = [(i - 1, 0)]
        if first_input is not None:
            self.inputs[old_m] = [(self.modules.index(first_input), 0)]

    def feed_model_input(self, module, ind=0):
        self.inputs[self._find_or_add(module)] = [(-1, ind)]

    def add_multi_input_layer(self, module, inputs):
        self.inputs[self._find_or_add(module)] = [(self.modules.index(m), 0) for m in inputs]

    def fan_out(self, module, outputs):
        mi = self.modules.index(module)
        for i, m in enumerate(outputs):
            self.inputs[self._find_or_add(m)] = [(mi, i)]

    def replicate_output(self, module, outputs):
        mi = self.modules.index(module)
        for m in outputs:
            self.inputs[self._find_or_add(m)] = [(mi, 0)]

    def compute_output_users(self):
        m = len(self.modules)
        self.output_users = [[] for _ in range(m)]
        self.model_input_users = []
        for i, inp in enumerate(self.inputs):
            for j, inp_item in enumerate(inp):
                if inp_item[0] >= 0:
                    self.output_users[inp_item[0]].append((i, j, inp_item[1]))
                else:
                    self.model_input_users.append((i, j, inp_item[1]))


class DistributedPipelineRecord:
    ready_cv: Condition
    batches: List[Batch]

    def __init__(self, device, rank, n, num_input, users):
        self.ready_cv = Condition()
        self.tensors = [[None] * num_input for _ in range(n)]
        self.batches = [None] * n
        self.batch_events = [[None] * num_input for _ in range(n)]
        self.forwarded_phony = [[[] for j in range(num_input)] for i in range(n)]
        self.users = users
        self.rank = rank
        self.device = device

    def feed(self, i: int, j: int, input):
        if input.device.type == "cpu":
            input = input.to(self.device)
        cuda_stream = torch.cuda.current_stream(input.device) if input.device.type == "cuda" else None

        with self.ready_cv:
            assert self.tensors[i][j] is None
            input, phony = fork(input)
            self.batch_events[i][j] = cuda_stream.record_event() if cuda_stream is not None else None
            self.tensors[i][j] = input
            self.ready_cv.notify_all()
        return phony

    def wait_for(self, chunk: int):
        with self.ready_cv:
            while self.batches[chunk] is None and any(b is None for b in self.tensors[chunk]):
                self.ready_cv.wait()
            if self.batches[chunk] is None:
                self.batches[chunk] = Batch(self.tensors[chunk], chunk)


class PartitionHandler:
    def __init__(
        self, module_rref, device, num_input, num_output, rank, chunks, checkpoint_stop: int, loss_module_rref=None
    ) -> None:
        self.module = module_rref.local_value()
        self.chunks = chunks
        self.device = torch.device(device)
        self.checkpoint_stop = checkpoint_stop
        self.rank = rank
        self.num_input = num_input
        self.num_output = num_output
        (self.in_queue,), (self.out_queue,) = create_workers([self.device])
        self.loss_module = None if loss_module_rref is None else loss_module_rref.local_value()

    def local_parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [rpc.RRef(p) for p in self.module.parameters()]

    def make_dist_record(self, users):
        return DistributedPipelineRecord(self.device, self.rank, self.chunks, self.num_input, users)

    def run(self, dist_record: DistributedPipelineRecord) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """

        m = len(dist_record.batches)

        self.stream = current_stream(self.device)

        for i in range(m):
            with record_function("feed"):
                dist_record.wait_for(i)
            self.fence(dist_record, i)
            self.compute(dist_record, i)
            self.forward_results(i, dist_record)

    def fence(self, dist_record: Optional[DistributedPipelineRecord], chunk,) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        # Ensure that batches[chunk-1] is executed after batches[chunk] in
        # backpropagation by an explicit dependency.
        if chunk != 0 and dist_record.users and dist_record.rank > 1:
            t = []
            for b, remote_ph_list in zip(dist_record.batches[chunk].tensors, dist_record.forwarded_phony[chunk - 1]):
                r = b
                for remote_ph in remote_ph_list:
                    ph = remote_ph.to_here()
                    r = join(r, ph)
                t.append(r)
            dist_record.batches[chunk] = Batch(t, chunk)

    def compute(self, dist_record: Optional[DistributedPipelineRecord], chunk) -> None:
        """Runs tasks with synchronization to copy streams."""
        checkpoint_stop = self.checkpoint_stop

        # Disable checkpointing if in eval mode.
        if not self.module.training:
            checkpoint_stop = 0

        exc_info: Optional[ExcInfo] = None

        batch = dist_record.batches[chunk]

        # Synchronize with the copied input. ([1] in the diagram)
        if dist_record is not None and dist_record.rank >= 0:
            for e in dist_record.batch_events[chunk]:
                if e is not None and is_cuda(self.stream):
                    self.stream.wait_event(e)

        # Determine whether checkpointing or not.
        checkpoint = chunk < checkpoint_stop
        if checkpoint:

            def function(input: TensorOrTensors, chunk_id: int = chunk,) -> TensorOrTensors:
                with record_function("chunk%d-rank%d" % (chunk_id, dist_record.rank)):
                    result = self.module(*input)
                    if self.num_output is None:
                        result = (result,)
                    return tuple(result)

            chk = Checkpointing(function, batch)
            task = Task(self.stream, compute=chk.checkpoint, finalize=chk.recompute)
            del function, chk

        else:

            def compute(
                batch: Batch = batch, chunk_id: int = chunk, rank=dist_record.rank if dist_record is not None else -1
            ) -> Batch:
                with record_function("chunk%d-rank%d" % (chunk_id, dist_record.rank)):
                    result = self.module(*batch.tensors)
                    if self.num_output is None:
                        result = (result,)
                return Batch(result, chunk_id)

            task = Task(self.stream, compute=compute, finalize=None)
            del compute

        # Compute tasks in parallel. ([2] in the diagram)
        self.in_queue.put(task)

        ok, payload = self.out_queue.get()

        # Hold the first exception.
        if exc_info is not None:
            pass
        elif not ok:
            exc_info = cast(ExcInfo, payload)
        else:
            task, batch = cast(Tuple[Task, Batch], payload)

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(self.device):
                task.finalize(batch)

            dist_record.batches[chunk] = batch

        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    def forward_results(self, chunk, dist_record: DistributedPipelineRecord):
        with use_stream(self.stream):
            for user, input_idx, output_idx in dist_record.users:
                v = dist_record.batches[chunk].value[output_idx]
                if BOUNCE_TENSORS:
                    v = v.cpu()
                dist_record.forwarded_phony[chunk][output_idx].append(user.remote().feed(chunk, input_idx, v))

    def run_pipeline(self, dist_record_ref):
        dist_record = dist_record_ref.local_value()
        self.run(dist_record)

        if not dist_record.users:
            result = microbatch.gather(dist_record.batches)
            assert len(result) == 1
            result = result[0]
            s0 = current_stream(result.device)
            if is_cuda(s0):
                s0.synchronize()
            return result


class MultiInputSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, *inputs):
        input = self.modules[0](*inputs)
        for module in self.modules[1:]:
            input = module(input)
        return input


def RemoteSequential(rref_list):
    return MultiInputSequential(*(r.local_value() for r in rref_list))


def _split_module(seq: RemoteModuleSequence):
    seq.compute_output_users()
    module_used = [False] * len(seq.modules)
    partitions = []
    for i, module in enumerate(seq.modules):
        if module_used[i]:
            continue
        partition = []
        j = i
        current_module = module
        while True:
            assert not module_used[j]
            module_used[j] = True
            partition.append(j)
            if len(seq.output_users[j]) != 1:
                break
            if seq.modules[j].num_outputs is not None:
                break
            next_j = seq.output_users[j][0][0]
            next_module = seq.modules[next_j]
            if seq.inputs[next_j] != [(j, 0)]:
                break
            if next_module.on != current_module.on:
                break
            if next_module.device != current_module.device:
                break
            current_module = next_module
            j = next_j
        if len(partition) == 1:
            remote_module = seq.modules[partition[0]].get_module_rref()
        else:
            remote_module = rpc.remote(
                seq.modules[partition[0]].on,
                RemoteSequential,
                args=([seq.modules[p].get_module_rref() for p in partition],),
            )
        partitions.append((partition, remote_module))

    return partitions


MOVING_DENIED = TypeError(
    "denied to move parameters and buffers, " "because DistributedPipeline should manage device placement"
)


class DistributedPipeline(nn.Module):
    def __init__(
        self, seq, chunks: int = 1, checkpoint: str = "except_last", deferred_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        chunks = int(chunks)
        checkpoint = str(checkpoint)

        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        self.chunks = chunks

        # if deferred_batch_norm:
        #    module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)

        self.partitions = _split_module(seq)
        self.input_feeds = [
            next((i, fj, feed_idx) for i, (p, m) in enumerate(self.partitions) if p[0] == fi)
            for fi, fj, feed_idx in seq.model_input_users
        ]

        self._copy_streams: List[List[AbstractStream]] = []

        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[checkpoint]

        self.partition_handlers = [
            rpc.remote(
                m.owner(),
                PartitionHandler,
                args=(
                    m,
                    seq.modules[p[0]].device,
                    seq.modules[p[0]].num_inputs,
                    seq.modules[p[-1]].num_outputs,
                    i,
                    self.chunks,
                    checkpoint_stop,
                ),
            )
            for i, (p, m) in enumerate(self.partitions)
        ]
        self.modules_sequence = seq

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self.partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions = self.partitions
        if index < 0:
            partitions = partitions[::-1]

        for partition in partitions:
            try:
                return partition[index]
            except IndexError:
                pass

            shift = len(partition)

            if index < 0:
                index += shift
            else:
                index -= shift

        raise IndexError

    def __iter__(self) -> Iterable[nn.Module]:
        """Iterates over children of the underlying sequential module."""
        for partition in self.partitions:
            yield from partition

    # DistributedPipeline should manage the device of each partition.
    # Deny cuda(), cpu(), and to() with device, by TypeError.
    def cuda(self, device: Optional[Device] = None) -> "DistributedPipeline":
        raise MOVING_DENIED

    def cpu(self) -> "DistributedPipeline":
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) -> "DistributedPipeline":
        # Deny these usages:
        #
        # - to(device[, dtype, non_blocking])
        # - to(tensor[, non_blocking])
        #
        # But allow this:
        #
        # - to(dtype[, non_blocking])
        #
        if "device" in kwargs or "tensor" in kwargs:
            raise MOVING_DENIED

        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED

        return super().to(*args, **kwargs)

    def parameter_rrefs(self):
        remote_params = []
        for p in self.partition_handlers:
            remote_params.extend(p.rpc_sync().local_parameter_rrefs())
        print("params rrefs: ", len(remote_params))
        return remote_params

    def forward(self, *inputs) -> rpc.RRef:  # type: ignore
        inputs = list(inputs)
        for i, input in enumerate(inputs):
            microbatch.check(input)
            # if input.device.type == 'cpu':
            #    inputs[i] = input.to(0)

        if not self.partition_handlers:
            # Empty sequential module is not illegal.
            assert len(inputs) == 1
            return rpc.RRef(inputs[0])

        # Divide a mini-batch into micro-batches.
        batches_list = [microbatch.scatter(input, self.chunks) for input in inputs]

        m = len(self.partition_handlers)
        dist_records = [None] * (m + 1)
        for part_idx in reversed(range(m)):
            rmodel = self.partition_handlers[part_idx].remote()
            users = []
            for user, input_idx, output_idx in self.modules_sequence.output_users[self.partitions[part_idx][0][-1]]:
                user_partition = next(i for i, (p, m) in enumerate(self.partitions) if p[0] == user)
                assert user_partition > part_idx
                users.append((dist_records[user_partition], input_idx, output_idx))
            dist_records[part_idx] = rmodel.make_dist_record(users)
            this_result = rmodel.run_pipeline(dist_records[part_idx])
            if part_idx == m - 1:
                result = this_result

        for i, b in enumerate(zip(*batches_list)):
            for fi, fj, feed_idx in self.input_feeds:
                # TODO: Debug why we need this special handling
                if dist_records[fi].owner().name == rpc.get_worker_info().name:
                    dist_records[fi].local_value().feed(i, fj, b[feed_idx].value)
                else:
                    dist_records[fi].rpc_async().feed(i, fj, b[feed_idx].value)

        return result
