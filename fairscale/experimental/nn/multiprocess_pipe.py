# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from threading import Condition
from types import TracebackType
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule

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


ConsumerType = TypeVar("ConsumerType")


@dataclass
class DataConsumer(Generic[ConsumerType]):
    """A data class for representating a consumer of an output of a module."""

    consumer: ConsumerType
    consumer_input_idx: int  # indicating which input of the consumer module
    output_idx: int  # indicating which output of the producer module


class PipelineModulesGraph(nn.Module):
    """A collection of remote modules (of type RemoteModule) with connections showing how inputs
    to the model or outputs of individual modules are use as inputs of subsequent modules.
    The graph has a number of helper functions that add new modules to the graph and define inputs
    to these module.
    """

    DataConsumer = DataConsumer["PipelineModulesGraph.Node"]

    @dataclass
    class DataSource:
        # If producer is None, we are referring to the model input
        producer: Optional["PipelineModulesGraph.Node"]
        # indicating which output of the producer module, or which input of the model if producer is None.
        output_idx: int

    class Node:
        def __init__(self, module: RemoteModule):
            self.module = module
            self.num_outputs: Optional[int] = None
            # self.inputs specifies inputs to the module. Each input is represented by a tuple (i, j).
            # If i>=0, then the input is the j'th output of the i'th # module in the graph. If i<0,
            # the input is the j'th input to the model.
            self.inputs: List["PipelineModulesGraph.DataSource"] = []
            # To be compiled by _compile method
            self.output_consumers: List[DataConsumer] = []

    def __init__(self) -> None:
        super().__init__()
        self.nodes: List[PipelineModulesGraph.Node] = []

    def _find_node(self, module: RemoteModule) -> Node:
        for n in self.nodes:
            if n.module is module:
                return n
        raise ValueError

    def _find_or_add(self, module: RemoteModule) -> Node:
        try:
            return self._find_node(module)
        except ValueError:
            new_node = self.Node(module)
            self.nodes.append(new_node)
            return new_node

    def _set_inputs(self, module: RemoteModule, inputs: List[DataSource]) -> None:
        self._find_or_add(module).inputs = inputs

    def add_sequence(self, modules: List[RemoteModule], first_input: Optional[RemoteModule] = None) -> None:
        """Adds a list of modules to the graph, to be run sequentially.
        The connection between these modules is as follows: the first output of each of these modules
        (except the last one) is used as the first input of its next module in this sequence.
        The user may also specify the input to the first module in this sequence with argument 'first_input'.
        In this case the module 'first_input' must have been added to the graph previously.
        """
        old_modules_len = len(self.nodes)
        new_modules_len = len(modules)
        self.nodes.extend(self.Node(mod) for mod in modules)
        # update inputs array
        if first_input is not None:
            self.nodes[old_modules_len].inputs = [self.DataSource(self._find_node(first_input), 0)]
        for i in range(old_modules_len + 1, old_modules_len + new_modules_len):
            self.nodes[i].inputs = [self.DataSource(self.nodes[i - 1], 0)]

    def set_model_input(self, module: RemoteModule, ind: int = 0) -> None:
        """Declares the input to a module as the input to the model. In case the model has multiple
        inputs, the argument 'ind' indicates the index of the model input that is fed to the module.
        """
        self._set_inputs(module, [self.DataSource(None, ind)])

    def add_multi_input_layer(self, module: RemoteModule, inputs: List[RemoteModule]) -> None:
        """Adds a module with multiple inputs to the graph. The modules that provide inputs to this module
        must have been added previously to the graph and are listed with argument inputs.
        """
        self._set_inputs(module, [self.DataSource(self._find_node(m), 0) for m in inputs])

    def fan_out(self, module: RemoteModule, outputs: List[RemoteModule]) -> None:
        """Feeds outputs of a previously added module to modules specified by argument 'outputs' (so
        'module' should have at least 'len(outputs)' outputs.
        Modules in the list 'outputs' are added to the graph if they have not been added previously.
        """
        node = self._find_node(module)
        node.num_outputs = len(outputs)
        for i, m in enumerate(outputs):
            self._set_inputs(m, [self.DataSource(node, i)])

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

        m = len(self.nodes)
        self.model_input_consumers = []
        for node in self.nodes:
            for input_index, input_item in enumerate(node.inputs):
                data_consumer = DataConsumer(node, input_index, input_item.output_idx)
                if input_item.producer is not None:
                    input_item.producer.output_consumers.append(data_consumer)
                else:
                    self.model_input_consumers.append(data_consumer)

    def _trace_modules(self, node: "PipelineModulesGraph.Node") -> List["PipelineModulesGraph.Node"]:
        """Compiles a list of modules (starting from module number module_idx), where each module in the list
        gets the output of previous module in the list as its input. So every module in the list, except the
        first one should have only one input, and similarly, every module in the list, except the last one
        should have only one output.
        """
        partition = []
        current_node = node
        while True:
            partition.append(current_node)
            # If we reached a module with multiple outputs or with multiple consumers for its output,
            # stop adding more modules to the partition.
            if len(current_node.output_consumers) != 1:
                break
            if current_node.num_outputs is not None:
                break
            # Next module to add is the only consumer of the ouput of the current module
            next_node = current_node.output_consumers[0].consumer
            # If the next module has multiple inputs, do not add it to the current partition and stop.
            if next_node.inputs != [(current_node, 0)]:
                break
            # If the next module is on a different deivce or worker, stop
            if next_node.module.on != current_node.module.on:
                break
            if next_node.module.device != current_node.module.device:
                break
            current_node = next_node

        return partition

    def partition_graph(self) -> List[Tuple[List[Node], rpc.RRef]]:
        """Splits the graph into pipeline partitions and for each parition returns a tuple (indices, module_rref),
        where indices is indices of modules of the partition in the graph, and module_rref is an RRef to an nn.Module:
        Each partition is a list of modules on the same device that are executed sequentially (output of each module is
        the input to the next module).
        If there is only one module in the partition, module_rref is reference to that module; otherwise those modules
        are wrapped by a MultiInputSequential and module_rref referes to that.
        """
        self._compile()
        modules_used: Set[PipelineModulesGraph.Node] = set()
        partitions = []
        for node in self.nodes:
            if node in modules_used:
                continue
            partition = self._trace_modules(node)
            assert not modules_used.intersection(partition)
            modules_used.update(partition)

            if len(partition) == 1:
                remote_module = partition[0].module.get_module_rref()
            else:
                remote_module = rpc.remote(
                    partition[0].module.on, RemoteSequential, args=([p.module.get_module_rref() for p in partition],),
                )
            partitions.append((partition, remote_module))

        return partitions


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

    DataConsumer = DataConsumer[rpc.RRef]

    def __init__(
        self,
        device: torch.device,
        rank: int,
        chunks: int,
        num_inputs: int,
        num_outputs: Optional[int],
        consumers: List[DataConsumer],
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

    def fence(self, chunk: int) -> None:
        """Prepares micro-batches for computation."""
        # Ensure that batches[chunk-1] is executed after batches[chunk] in
        # backpropagation by an explicit dependency.
        # TODO: This dependency injection causes deadlock if this partition
        # gets its input from model input. 1) Figure out why 2) If we need to live
        # with this constraint, replace the condition 'self.rank > 0' below with
        # a more accurate one.
        if chunk != 0 and self.consumers and self.rank > 0:
            dependant_tensors = []
            batch = self.batches[chunk]
            assert batch is not None
            for tensor, remote_ph_list in zip(batch.tensors, self.forwarded_phony[chunk - 1]):
                dependant = tensor
                for remote_ph in remote_ph_list:
                    phony = remote_ph.to_here()
                    dependant = join(dependant, phony)
                dependant_tensors.append(dependant)
            self.batches[chunk] = Batch(tuple(dependant_tensors), chunk)

    def sync_stream(self, chunk: int, stream: torch.cuda.Stream) -> None:
        """syncs the stream with cuda events associated with transmission of the chunck to the cuda device."""
        for e in self.recv_events[chunk]:
            if e is not None:
                stream.wait_event(e)

    def forward_results(self, chunk: int) -> None:
        """Forward outputs of processing the chunk in this parition for processing by next partition."""
        for consumer in self.consumers:
            v = self.get_batch(chunk).value[consumer.output_idx]
            self.forwarded_phony[chunk][consumer.output_idx].append(
                consumer.consumer.remote().feed(chunk, consumer.consumer_input_idx, v)
            )

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

    def make_pipeline_record(
        self, consumers: List[DistributedPipelineRecord.DataConsumer]
    ) -> DistributedPipelineRecord:
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
            pipeline_record.fence(chunk)
            self.compute(pipeline_record, chunk)
            with use_stream(self.stream):
                pipeline_record.forward_results(chunk)

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

    @dataclass
    class Partition:
        nodes: List[PipelineModulesGraph.Node]
        handler: rpc.RRef

        def __hash__(self) -> int:
            return hash(self.handler)

    DataConsumer = DataConsumer[Partition]

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
        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[checkpoint]

        self.partitions = [
            self.Partition(
                nodes,
                rpc.remote(
                    handler.owner(),
                    PartitionHandler,
                    args=(
                        handler,
                        nodes[0].module.device,
                        len(nodes[0].inputs),
                        nodes[-1].num_outputs,
                        i,
                        self.chunks,
                        checkpoint_stop,
                    ),
                ),
            )
            for i, (nodes, handler) in enumerate(graph.partition_graph())
        ]
        self.input_consumers = [
            next(
                self.DataConsumer(partition, input_consumer.consumer_input_idx, input_consumer.output_idx)
                for partition in self.partitions
                if partition.nodes[0] is input_consumer.consumer
            )
            for input_consumer in graph.model_input_consumers
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
        for p in self.partitions:
            remote_params.extend(p.handler.rpc_sync().local_parameter_rrefs())
        return remote_params

    def forward(self, *inputs: Tensor) -> rpc.RRef:  # type: ignore
        for i, input in enumerate(inputs):
            microbatch.check(input)

        # Divide a mini-batch into micro-batches.
        batches_list = [microbatch.scatter(input, self.chunks) for input in inputs]
        num_partitions = len(self.partitions)

        # Create a DistributedPipelineRecord, one per partition, and make connections between them (i.e.
        # set list of consumers).
        pipeline_records: Dict[DistributedPipeline.Partition, rpc.RRef] = {}
        for partition in reversed(self.partitions):
            r_handler = partition.handler.remote()
            consumers = []
            # Identify consumers of the outputs of the partition
            for consumer in partition.nodes[-1].output_consumers:
                consumer_partition = next(p for p in self.partitions if p.nodes[0] is consumer.consumer)
                # Index of a consumer partition should be greater than index of the partition.
                assert consumer_partition in pipeline_records
                consumers.append(
                    DistributedPipelineRecord.DataConsumer(
                        pipeline_records[consumer_partition], consumer.consumer_input_idx, consumer.output_idx
                    )
                )
            pipeline_records[partition] = r_handler.make_pipeline_record(consumers)
            # Let the pipeline-handler for the partition starts processing the pipeline-record for that partition.
            this_result = r_handler.run_pipeline(pipeline_records[partition])
            # If this is the last partition, we expect the result of the model be the output of this partition.
            if partition is self.partitions[-1]:
                result = this_result

        # Start feeding model input to the partitions that need them.
        for i, b in enumerate(zip(*batches_list)):
            for input_consumer in self.input_consumers:
                pipeline_record = pipeline_records[input_consumer.consumer]
                # TODO: Debug why we need this special handling
                if pipeline_record.owner().name == rpc.get_worker_info().name:  # type: ignore
                    pipeline_record.local_value().feed(
                        i, input_consumer.consumer_input_idx, b[input_consumer.output_idx].value
                    )
                else:
                    pipeline_record.rpc_async().feed(i, input_consumer.consumer_input_idx, b[input_consumer.output_idx].value)  # type: ignore

        return result
