# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from threading import Condition
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed import rpc

from fairscale.nn.pipe import microbatch
from fairscale.nn.pipe.checkpoint import Checkpointing, TensorOrTensors
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.stream import as_cuda, current_stream, is_cuda, use_device, use_stream
from fairscale.nn.pipe.worker import Task, create_workers

from .data import DataConsumer

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


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

    # Need to use Union due to https://github.com/python/mypy/issues/7866
    DataConsumer = Union[DataConsumer[rpc.RRef]]

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

    def make_pipeline_record(self, consumers: List[DataConsumer]) -> DistributedPipelineRecord:
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
