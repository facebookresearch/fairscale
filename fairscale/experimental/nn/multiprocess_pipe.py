# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from threading import Condition
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import rpc

from fairscale.nn.pipe import microbatch
from fairscale.nn.pipe.checkpoint import Checkpointing, TensorOrTensors
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.stream import as_cuda, current_stream, is_cuda, use_device, use_stream
from fairscale.nn.pipe.worker import Task, create_workers

Device = Union[torch.device, int, str]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


def check_pytorch_version() -> None:
    if torch.__version__.split("+")[0].split(".")[:2] < ["1", "9"]:
        raise Exception("DistributedPipeline requires PyTorch version 1.9 or higher")


def _rloss(loss_func: Callable, input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
    return loss_func(input_rref.to_here(), target_rref.to_here())


def DistributedLoss(loss: nn.Module, *args: Tuple, **kwargs: Dict) -> Callable:
    loss_func = loss(*args, **kwargs)

    def dloss(input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
        return rpc.remote(input_rref.owner(), _rloss, args=(loss_func, input_rref, target_rref))

    return dloss


class MultiInputSequential(nn.Module):
    """A variation of nn.Sequential, that allows the first module in the sequence accepts
        multiple inputs. To be used internally by _split_module
    """

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.modules_list = modules

    def forward(self, *inputs: Tuple[Tensor]) -> Tensor:  # type: ignore
        input = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            input = module(input)
        return input


def RemoteSequential(rref_list: List[rpc.RRef]) -> MultiInputSequential:
    return MultiInputSequential(*(r.local_value() for r in rref_list))


class PipelineModule(nn.Module):
    """Constructs a module on a remote device, possibly at a later time (in case the device is not
    specified when creating PipelineModule.
    Args:
        module_cls (nn.Module): Class for the module to be created remotely.
        args (Sequence): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.
        num_inputs (int, optional): number of inputs to the forward function.
        num_outputs: (int, optional): If the forward function returns a tuple, number of elements
            in the tuple, otherwise it should be None
        remote_device: (str, optional): Device on the destination worker where we‘d like to place
            this module. The format should be "<workername>/<device>", where the device field can be
            parsed as torch.device type. E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            This parameter can be provided later by calling the method instantiate.
    """

    def __init__(
        self,
        module_cls: nn.Module,
        args: Tuple,
        kwargs: Optional[Dict] = None,
        num_inputs: int = 1,
        num_outputs: Optional[int] = None,
        remote_device: str = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.module_args = (module_cls, args, kwargs or {})
        if remote_device is not None:
            self.instantiate(remote_device)

    @staticmethod
    def _create_module(module_cls: Callable, args: Tuple, kwargs: Dict, device: str) -> nn.Module:
        result: nn.Module = module_cls(*args, **kwargs)
        result.to(device)
        return result

    def instantiate(self, remote_device: str) -> "PipelineModule":
        worker, device = remote_device.split("/")
        self.worker = worker
        self.device = device
        self.module_rref = rpc.remote(worker, PipelineModule._create_module, self.module_args + (device,))
        return self

    def get_module_rref(self) -> rpc.RRef:
        return self.module_rref


class PipelineModulesGraph(nn.Module):
    """A collection of remote modules (of type PipelineModule) with connections showing how inputs
    to the model or outputs of individual modules are use as inputs of subsequent modules.
    The graph has a number of helper functions that add new modules to the graph and define inputs
    to these module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.modules_list: List = []
        # self.inputs specifies inputs to each module in modules_list. Each input to each module
        # is represented by a tuple (i, j). If i>=0, then the input is the j'th output of the i'th
        # module. If i<0, the input is the j'th input to the model.
        self.inputs: List[Optional[List[Tuple[int, int]]]] = []

    def _find_or_add(self, module: PipelineModule) -> int:
        try:
            return self.modules_list.index(module)
        except ValueError:
            self.inputs.append(None)
            self.modules_list.append(module)
            return len(self.modules_list) - 1

    def _set_inputs(self, module: PipelineModule, inputs: List[Tuple[int, int]]) -> None:
        self.inputs[self._find_or_add(module)] = inputs

    def add_sequence(self, modules: List[PipelineModule], first_input: Optional[PipelineModule] = None) -> None:
        """Adds a list of modules to the graph, to be run sequentially.
        The connection between these modules is as follows: the first output of each of these modules
        (except the last one) is used as the first input of its next module in this sequence.
        The user may also specify the input to the first module in this sequence with argument 'first_input'.
        In this case the module 'first_input' must have been added to the graph previously.
        """
        old_modules_len = len(self.modules_list)
        new_modules_len = len(modules)
        self.modules_list.extend(modules)
        # update inputs array
        self.inputs.append([(self.modules_list.index(first_input), 0)] if first_input is not None else None)
        for i in range(old_modules_len + 1, old_modules_len + new_modules_len):
            self.inputs.append([(i - 1, 0)])

    def set_model_input(self, module: PipelineModule, ind: int = 0) -> None:
        """Declares the input to a module as the input to the model. In case the model has multiple
        inputs, the argument 'ind' indicates the index of the model input that is fed to the module.
        """
        self._set_inputs(module, [(-1, ind)])

    def add_multi_input_layer(self, module: PipelineModule, inputs: List[PipelineModule]) -> None:
        """Adds a module with multiple inputs to the graph. The modules that provide inputs to this module
        must have been added previously to the graph and are listed with argument inputs.
        """
        self._set_inputs(module, [(self.modules_list.index(m), 0) for m in inputs])

    def fan_out(self, module: PipelineModule, outputs: List[PipelineModule]) -> None:
        """Feeds outputs of a previously added module to modules specified by argument 'outputs' (so
        'module' should have at least 'len(outputs)' outputs.
        Modules in the list 'outputs' are added to the graph if they have not been added previously.
        """
        mi = self.modules_list.index(module)
        for i, m in enumerate(outputs):
            self._set_inputs(m, [(mi, i)])

    @dataclass
    class OutputConsumerIndex:
        """A data class for representating a consumer of an output of a module."""

        consumer_idx: int  # index of the consumer module
        consumer_input_idx: int  # indicating which input of the consumer module
        producer_output_idx: int  # indicating which output of the producer module

    @staticmethod
    def _is_model_input(input_desc: Tuple[int, int]) -> bool:
        """Checks if an entry in self.inputs refers to a model input"""
        return input_desc[0] < 0

    def _compile(self) -> None:
        """Precomputes self.model_input_consumers and self.output_consumers for internal use by the pipleine
        class. These two lists show consumers of inputs to the model, and outputs of each module of
        the graph. Each consumer is a pair (i, j) which stands for the j'th input to the i'th module
        in the graph.
        """

        # TODO: We need to make sure following conditions hold before preparing the graph for the pipeline:
        #   * the graph has a least one module, and is connected.
        #   * num_inputs and num_outputs for modules matche list of connections defined in the graph.
        #   * all inputs to a module should come from model input, or modules with smaller index in
        #     the graph. This condition is used in implementaion of DistributedPipeline.forward. Even
        #     if we relax this condition, still need to make sure the graph is acyclic.

        m = len(self.modules_list)
        self.output_consumers: List[List[PipelineModulesGraph.OutputConsumerIndex]] = [[] for _ in range(m)]
        self.model_input_consumers = []
        for i, input in enumerate(self.inputs):
            assert input is not None
            for j, input_item in enumerate(input):
                if not self._is_model_input(input_item):
                    self.output_consumers[input_item[0]].append(
                        PipelineModulesGraph.OutputConsumerIndex(i, j, input_item[1])
                    )
                else:
                    self.model_input_consumers.append((i, j, input_item[1]))

    def _trace_modules(self, module_idx: int) -> List[int]:
        """Compiles a list of modules (starting from module number module_idx), where each module in the list
        gets the output of previous module in the list as its input. So every module in the list, except the
        first one should have only one input, and similarly, every module in the list, except the last one
        should have only one output.
        """
        partition = []
        current_module_idx = module_idx
        current_module = self.modules_list[module_idx]
        while True:
            partition.append(current_module_idx)
            # If we reached a module with multiple outputs or with multiple consumers for its output,
            # stop adding more modules to the partition.
            if len(self.output_consumers[current_module_idx]) != 1:
                break
            if self.modules_list[current_module_idx].num_outputs is not None:
                break
            # Next module to add is the only consumer of the ouput of the current module
            next_module_idx = self.output_consumers[current_module_idx][0].consumer_idx
            next_module = self.modules_list[next_module_idx]
            # If the next module has multiple inputs, do not add it to the current partition and stop.
            if self.inputs[next_module_idx] != [(current_module_idx, 0)]:
                break
            # If the next module is on a different deivce or worker, stop
            if next_module.worker != current_module.worker:
                break
            if next_module.device != current_module.device:
                break
            current_module = next_module
            current_module_idx = next_module_idx

        return partition

    def partition_graph(self) -> List[Tuple[List[int], rpc.RRef]]:
        """Splits the graph into pipeline partitions and for each parition returns a tuple (indices, module_rref),
        where indices is indices of modules of the partition in the graph, and module_rref is an RRef to an nn.Module:
        Each partition is a list of modules on the same device that are executed sequentially (output of each module is
        the input to the next module).
        If there is only one module in the partition, module_rref is reference to that module; otherwise those modules
        are wrapped by a MultiInputSequential and module_rref referes to that.
        """
        self._compile()
        module_used = [False] * len(self.modules_list)
        partitions = []
        for module_idx, module in enumerate(self.modules_list):
            if module_used[module_idx]:
                continue
            partition = self._trace_modules(module_idx)
            for idx in partition:
                assert not module_used[idx]
                module_used[idx] = True

            if len(partition) == 1:
                remote_module = self.modules_list[partition[0]].get_module_rref()
            else:
                remote_module = rpc.remote(
                    self.modules_list[partition[0]].worker,
                    RemoteSequential,
                    args=([self.modules_list[p].get_module_rref() for p in partition],),
                )
            partitions.append((partition, remote_module))

        return partitions


@dataclass
class OutputConsumer:
    """A data class for representating a consumer of an output of the module."""

    consumer_rref: rpc.RRef  # reference to the consumer (of type DistributedPipelineRecord)
    consumer_input_idx: int  # indicating which input of the consumer module
    output_idx: int  # indicating which output of the module


class DistributedPipelineRecord:
    """ A class for storing a single mini-batch (consisting of multiple micro-batches) as input to
    a single partition.
    Args:
        device: the local device that runs the partition.
        rank: the rank of the partition in the pipeline.
        chunks: number of micro-batches in a mini-batch
        num_inputs: number of inputs to the partition.
        consumers: list of consumers of outputs of the partition. Each consumer in the list is a tuple
            (remote_partition_rref, input_idx, output_idx) where remote_partition_rref points to a
            remote DistributedPipelineRecord for consumer partiton for this mini-batch. The output number
            output_idx of this partition will be used as the input number input_idx of that partition.
    """

    def __init__(
        self,
        device: torch.device,
        rank: int,
        chunks: int,
        num_inputs: int,
        num_outputs: Optional[int],
        consumers: List[OutputConsumer],
    ) -> None:
        self.ready_cv = Condition()
        # Each chunk consists of num_inputs tensors. self.tensors stores these individual tensors.
        self.tensors: List[List[Optional[Tensor]]] = [[None] * num_inputs for _ in range(chunks)]
        # For each tensor in self.tensors, we record a cuda event in corrsponding tensorpipe stream in self.recv_events,
        # and later the stream that processes that tensor will wait on that event.
        self.recv_events = [[None] * num_inputs for _ in range(chunks)]
        # Once all num_inputs tensors of a given chunk are recieved, they are assembled as a batch and stored in
        # self.batches
        self.batches: List[Optional[Batch]] = [None] * chunks
        # For each tensor of each chunk, we fork a phony tensor, which will be used for injecting dependency between
        # different chunks in backward path.
        if num_outputs is None:
            num_outputs = 1
        self.forwarded_phony: List[List[List[rpc.RRef]]] = [[[] for j in range(num_outputs)] for i in range(chunks)]
        self.consumers = consumers
        self.rank = rank
        self.device = device

    def feed(self, chunk: int, input_idx: int, input: Tensor) -> Tensor:
        """ This function is called remotely to provide individual tensors of a given chunk."""
        if input.device.type == "cpu":
            input = input.to(self.device)
        cuda_stream = torch.cuda.current_stream(input.device) if input.device.type == "cuda" else None

        with self.ready_cv:
            assert self.tensors[chunk][input_idx] is None
            input, phony = fork(input)
            self.recv_events[chunk][input_idx] = (
                cuda_stream.record_event() if cuda_stream is not None else None  # type: ignore
            )
            self.tensors[chunk][input_idx] = input
            self.ready_cv.notify_all()
        return phony

    def wait_for(self, chunk: int) -> None:
        """Waits until all elements of given chunk is populated in self.tensors.
        Then it constructs self.batches[chunk] if it is not constructed yet.
        """
        with self.ready_cv:
            while self.batches[chunk] is None and any(b is None for b in self.tensors[chunk]):
                self.ready_cv.wait()
            if self.batches[chunk] is None:
                tensors = cast(List[Tensor], self.tensors[chunk])
                self.batches[chunk] = Batch(tuple(tensors), chunk)

    def sync_stream(self, chunk: int, stream: torch.cuda.Stream) -> None:
        """syncs the stream with cuda events associated with transmission of the chunck to the cuda device."""
        for e in self.recv_events[chunk]:
            if e is not None:
                stream.wait_event(e)

    def get_batch(self, chunk: int) -> Batch:
        batch = self.batches[chunk]
        assert batch is not None
        return batch


class PartitionHandler:
    """This class processes a single partition of the pipeline.
    Args:
        module_rref: RRef to the nn.Module for this partition. It should be on the local rpc worker.
        device: The device that holds the module.
        num_inputs: Numer of inputs to the module
        num_outputs: Number of outputs of the module. If the module output is not a tuple (and it is a
            single tensor), num_outputs should be None.
        rank: The rank of the partition
        chunks: Number of micor-batches in a mini-batch
        checkpoint_stop:: Checkpointing is done only for the first checkpoint_stop chunks of a mini-batch.
    """

    def __init__(
        self,
        module_rref: rpc.RRef,
        device: str,
        num_inputs: int,
        num_outputs: Optional[int],
        rank: int,
        chunks: int,
        checkpoint_stop: int,
    ) -> None:
        self.module = module_rref.local_value()
        self.chunks = chunks
        self.device = torch.device(device)
        self.checkpoint_stop = checkpoint_stop
        self.rank = rank
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        (self.in_queue,), (self.out_queue,) = create_workers([self.device])

    def local_parameter_rrefs(self) -> List[rpc.RRef]:
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [rpc.RRef(p) for p in self.module.parameters()]

    def make_pipeline_record(self, consumers: List[OutputConsumer]) -> DistributedPipelineRecord:
        return DistributedPipelineRecord(
            self.device, self.rank, self.chunks, self.num_inputs, self.num_outputs, consumers
        )

    def run(self, pipeline_record: DistributedPipelineRecord) -> None:
        """Runs pipeline parallelism. It modifies the given batches in place."""
        m = len(pipeline_record.batches)

        self.stream = current_stream(self.device)

        for chunk in range(m):
            with record_function("feed"):
                pipeline_record.wait_for(chunk)
            self.fence(pipeline_record, chunk)
            self.compute(pipeline_record, chunk)
            # Forward outputs of processing the chunk in this parition for processing by next partition.
            with use_stream(self.stream):
                for consumer in pipeline_record.consumers:
                    v = pipeline_record.get_batch(chunk).value[consumer.output_idx]
                    pipeline_record.forwarded_phony[chunk][consumer.output_idx].append(
                        consumer.consumer_rref.remote().feed(chunk, consumer.consumer_input_idx, v)
                    )

    def fence(self, pipeline_record: DistributedPipelineRecord, chunk: int) -> None:
        """Prepares micro-batches for computation."""
        # Ensure that batches[chunk-1] is executed after batches[chunk] in
        # backpropagation by an explicit dependency.
        # TODO: This dependency injection causes deadlock if this partition
        # gets its input from model input. 1) Figure out why 2) If we need to live
        # with this constraint, replace the condition 'pipeline_record.rank > 0' below with
        # a more accurate one.
        if chunk != 0 and pipeline_record.consumers and pipeline_record.rank > 0:
            dependant_tensors = []
            batch = pipeline_record.batches[chunk]
            assert batch is not None
            for tensor, remote_ph_list in zip(batch.tensors, pipeline_record.forwarded_phony[chunk - 1]):
                dependant = tensor
                for remote_ph in remote_ph_list:
                    phony = remote_ph.to_here()
                    dependant = join(dependant, phony)
                dependant_tensors.append(dependant)
            pipeline_record.batches[chunk] = Batch(tuple(dependant_tensors), chunk)

    def compute(self, pipeline_record: DistributedPipelineRecord, chunk: int) -> None:
        """Runs tasks with synchronization to tensor-pipe streams."""
        checkpoint_stop = self.checkpoint_stop

        # Disable checkpointing if in eval mode.
        if not self.module.training:
            checkpoint_stop = 0

        exc_info: Optional[ExcInfo] = None

        batch = pipeline_record.get_batch(chunk)

        if is_cuda(self.stream):
            pipeline_record.sync_stream(chunk, as_cuda(self.stream))

        # Determine whether checkpointing or not.
        checkpoint = chunk < checkpoint_stop
        if checkpoint:

            def function(input: TensorOrTensors, chunk_id: int = chunk) -> TensorOrTensors:
                with record_function("chunk%d-rank%d" % (chunk_id, pipeline_record.rank)):
                    result = self.module(*input)
                    if self.num_outputs is None:
                        result = (result,)
                    return tuple(result)

            chk = Checkpointing(function, batch)
            task = Task(self.stream, compute=chk.checkpoint, finalize=chk.recompute)
            del function, chk

        else:

            def compute(
                batch: Batch = batch,
                chunk_id: int = chunk,
                rank: int = pipeline_record.rank if pipeline_record is not None else -1,
            ) -> Batch:
                with record_function("chunk%d-rank%d" % (chunk_id, pipeline_record.rank)):
                    result = self.module(*batch.tensors)
                    if self.num_outputs is None:
                        result = (result,)
                return Batch(result, chunk_id)

            task = Task(self.stream, compute=compute, finalize=None)
            del compute

        self.in_queue.put(task)

        ok, payload = self.out_queue.get()

        # Hold the first exception.
        if exc_info is not None:
            pass
        elif not ok:
            exc_info = cast(ExcInfo, payload)
        else:
            task, batch = cast(Tuple[Task, Batch], payload)

            with use_device(self.device):
                task.finalize(batch)

            pipeline_record.batches[chunk] = batch

        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    def run_pipeline(self, pipeline_record_rref: rpc.RRef) -> Optional[Tensor]:
        """Processes a min-batch on this partition.
           If this is the last partition (pipeline_record has no consumer), concatenates results of processing
           all chunks and returns the result as the output of the model on the whole mini-batch.
        """
        pipeline_record = pipeline_record_rref.local_value()
        self.run(pipeline_record)

        if not pipeline_record.consumers:
            result = microbatch.gather(pipeline_record.batches)
            assert len(result) == 1
            result = result[0]
            s0 = current_stream(result.device)
            if is_cuda(s0):
                # TODO. Investigate why this is needed and remove it if possible.
                as_cuda(s0).synchronize()
            return result

        return None


MOVING_DENIED = TypeError(
    "denied to move parameters and buffers, " "because DistributedPipeline should manage device placement"
)


class DistributedPipeline(nn.Module):
    """Wraps a :class:`PipelineModulesGraph` model to train on using synchronous pipeline
    parallelism. If the model requires lots of memory and doesn't fit on a single GPU,
    pipeline parallelism is a useful technique to employ for training.

    The implementation is based on the torchgpipe_ paper.

    .. _torchgpipe: https://arxiv.org/abs/2004.09910

    PipelineModulesGraph combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should place all the modules on the appropriate rpc workers and devices and wrap
    them into an :class:`PipelineModulesGraph` module defining the connection between them.

    Args:
        module (:class:`PipelineModulesGraph`):
        model to be parallelized using pipelining. Each module
            in the graph has to have all of its parameters on a single
            device.
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``).
            ``'never'`` disables checkpointing completely, ``'except_last'``
            enables checkpointing for all micro-batches except the last one
            and ``'always'`` enables checkpointing for all micro-batches.
    """

    def __init__(self, graph: PipelineModulesGraph, chunks: int = 1, checkpoint: str = "except_last",) -> None:
        super().__init__()

        check_pytorch_version()

        chunks = int(chunks)
        checkpoint = str(checkpoint)

        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        self.chunks = chunks

        self.partitions = graph.partition_graph()
        self.input_feeds = [
            next((i, fj, feed_idx) for i, (p, m) in enumerate(self.partitions) if p[0] == fi)
            for fi, fj, feed_idx in graph.model_input_consumers
        ]

        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[checkpoint]

        self.partition_handlers = [
            rpc.remote(
                m.owner(),
                PartitionHandler,
                args=(
                    m,
                    graph.modules_list[p[0]].device,
                    graph.modules_list[p[0]].num_inputs,
                    graph.modules_list[p[-1]].num_outputs,
                    i,
                    self.chunks,
                    checkpoint_stop,
                ),
            )
            for i, (p, m) in enumerate(self.partitions)
        ]
        self.graph = graph

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

    def parameter_rrefs(self) -> List[rpc.RRef]:
        remote_params = []
        for p in self.partition_handlers:
            remote_params.extend(p.rpc_sync().local_parameter_rrefs())
        return remote_params

    def forward(self, *inputs: Tensor) -> rpc.RRef:  # type: ignore
        for i, input in enumerate(inputs):
            microbatch.check(input)

        # Divide a mini-batch into micro-batches.
        batches_list = [microbatch.scatter(input, self.chunks) for input in inputs]
        num_partitions = len(self.partition_handlers)

        # Create a DistributedPipelineRecord, one per partition, and make connections between them (i.e.
        # set list of consumers).
        pipeline_records: List[Optional[rpc.RRef]] = [None] * (num_partitions + 1)
        for part_idx in reversed(range(num_partitions)):
            r_handler = self.partition_handlers[part_idx].remote()
            consumers = []
            # Identify consumers of the outputs of the partition
            for consumer in self.graph.output_consumers[self.partitions[part_idx][0][-1]]:
                consumer_partition_idx = next(
                    i for i, (p, num_partitions) in enumerate(self.partitions) if p[0] == consumer.consumer_idx
                )
                # Index of a consumer partition should be greater than index of the partition.
                assert consumer_partition_idx > part_idx
                consumer_partition = pipeline_records[consumer_partition_idx]
                assert consumer_partition is not None
                consumers.append(
                    OutputConsumer(consumer_partition, consumer.consumer_input_idx, consumer.producer_output_idx)
                )
            pipeline_records[part_idx] = r_handler.make_pipeline_record(consumers)
            # Let the pipeline-handler for the partition starts processing the pipeline-record for that partition.
            this_result = r_handler.run_pipeline(pipeline_records[part_idx])
            # If this is the last partition, we expect the result of the model be the output of this partition.
            if part_idx == num_partitions - 1:
                result = this_result

        # Start feeding model input to the partitions that need them.
        for i, b in enumerate(zip(*batches_list)):
            for fi, fj, feed_idx in self.input_feeds:
                pipeline_record = pipeline_records[fi]
                assert pipeline_record is not None
                # TODO: Debug why we need this special handling
                if pipeline_record.owner().name == rpc.get_worker_info().name:  # type: ignore
                    pipeline_record.local_value().feed(i, fj, b[feed_idx].value)
                else:
                    pipeline_record.rpc_async().feed(i, fj, b[feed_idx].value)  # type: ignore

        return result
