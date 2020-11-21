# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A nn.Module wrapper to go with a Sharded Optimizer in order to handle targeted gradient
reduction automatically.
"""

import contextlib
from itertools import chain
import logging
from typing import Any, Callable, Generator, List, Tuple, Union

import torch
from torch import nn
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim import OSS
from fairscale.optim.utils import Workhandle


class ShardedDataParallel(nn.Module):
    """
    Wrap the model, and reduce the gradients to the right rank during the backward pass.

    - the partition is given by the sharded optimizer
    - wrap the base model with a model which knows where to reduce each gradient
    - add an autograd function which calls the model grad dispatch on the way back

     Args:
        module (nn.Module):
            model to be wrapped
        sharded_optimizer (OSS, or list of OSS):
            the sharded optimizer(s) which will decide the gradient partitioning

    Keyword Args:
        process_group (torch.nn.Optimizer):
            Optimizer to shard (default: SGD)
        process_group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffers (bool):
            Whether to additionally broadcast model buffers in between ranks at the beginning of each forward pass.
            Same setting as in Pytorch DDP, this is in addition to the broadcast and reduction of the model parameters.
        sync_models_at_startup (bool):
            Synchronize the models in between the ranks when starting up. Not needed if each rank has the same seed,
            or the training restarts from a saved state

    """

    def __init__(
        self,
        module: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any = None,
        broadcast_buffers: bool = True,
        sync_models_at_startup: bool = True,
    ):
        super().__init__()

        self.module = module
        self.sharded_optimizers = [sharded_optimizer] if isinstance(sharded_optimizer, OSS) else sharded_optimizer
        self.enable_broadcast_buffers = broadcast_buffers

        # Handle a no_sync() context which prevents the gradient synchronization,
        # accumulate in place
        self.should_accumulate_grads = False

        # Communication related attributes
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.reference_global_rank = OSS.get_global_rank(self.process_group, 0)  # picking rank 0 as the reference
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = OSS.get_global_rank(self.process_group, self.rank)

        # Expose some of the PytorchDDP attributes, some frameworks rely on them.
        # See https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel
        # device_id related logic is not present, this is not handled
        devices = {p.device for p in self.module.parameters()}
        self.is_multi_device_module = len(devices) > 1
        self.device = list(devices)[0]

        distinct_device_types = {p.device.type for p in self.module.parameters()}
        assert len(distinct_device_types) == 1, (
            "ShardedDataParallel's input module must be on "
            "the same type of devices, but input module parameters are located on {} different device types."
        ).format(distinct_device_types)
        self.device_type = list(distinct_device_types)[0]

        # Scafolding to be able to reduce the grads during the BW pass
        # several optimizers can be present each working on seperate parameter sets,
        # we build an iterator which goes through all the parameters involved globally
        self._param_iterator = chain(*[optim.should_bucket_param.keys() for optim in self.sharded_optimizers])
        self._grad_to_be_reduced = [True for _ in self._param_iterator]
        self._grad_accs: List[Callable] = []
        self._setup_backward_hooks()

        # Make sure that all ranks start with the same model
        if sync_models_at_startup:
            self._sync_params_and_buffers()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Module forward pass, handles any DDP-specific work in the background. Primes the
        backward pass for gradient reduction to the proper ranks.
        """
        if self.enable_broadcast_buffers:
            # NCCL communications are on a different stream, needs to be blocking
            # for the subsequent FW to be correct
            self.sync_buffers(blocking=True)

        # Normal FW on the base model
        return self.module(*inputs, **kwargs)

    def reduce(self) -> None:
        """ .. deprecated:: 0.0.4

            This does not need to be called, the gradient reduction is done automatically during the BW pass
        """
        logging.warning("This is not useful anymore, gradients have been reduced automatically with the backward pass")

    def _sync_params_and_buffers(self) -> None:
        """
        Sync the complete model states in between the ranks
        """
        with torch.no_grad():
            work_handles = [
                dist.broadcast(t, src=self.reference_global_rank, group=self.process_group, async_op=True)
                for t in self.module.state_dict().values()
            ]

            _ = list(map(lambda x: x.wait(), work_handles))

    def sync_buffers(self, blocking: bool = False) -> None:
        """
        Sync all the param buffers in between ranks (including for instance batch norm statistics).
        """
        with torch.no_grad():
            work_handles = [
                dist.broadcast(buffer.data, self.reference_global_rank, self.process_group, async_op=True)
                for buffer in self.module.buffers(recurse=True)
            ]

            if blocking:
                _ = list(map(lambda x: x.wait(), work_handles))

    @contextlib.contextmanager
    def no_sync(self) -> Generator:
        """A context manager to disable gradient synchronization."""
        old_should_accumulate_grads = self.should_accumulate_grads
        self.should_accumulate_grads = True
        yield
        self.should_accumulate_grads = old_should_accumulate_grads

    def _find_rank(self, param: Parameter) -> Tuple[OSS, int]:
        """ Look up where this parameter belongs to """
        for optim in self.sharded_optimizers:
            if param in optim.param_to_rank.keys():
                return optim, optim.param_to_rank[param]

        assert False, "This parameter is not present in an optimizer, this should not happen"
        return (None, -1)

    def _get_reduce_fn(
        self, index: int, param: torch.Tensor, should_bucket: bool, dst_rank: int, optimizer: OSS
    ) -> Callable:
        """
        Two possible backward hooks for a given parameter: either directly reduce to the appropriate rank,
        or contribute to a bucket and reduce when the bucket is full.

        Either way a delayed action is necessary and is passed as a callback.
        """

        def gatekeeper() -> None:
            # Make sure that all the asynchronous calls have concluded before moving on. Consume the futures
            # and execute the delayed actions (release gradients, unroll the buckets)
            Variable._execution_engine.queue_callback(optimizer._consume_work_handles)

            # Reset all the grad reduce and bucket state flags
            self._grad_to_be_reduced = [True] * len(self._grad_to_be_reduced)

        def reduce_direct(*_: Any) -> None:
            # Skip gradient reduction, do not alter status flags
            if not self.should_accumulate_grads and self._grad_to_be_reduced[index]:
                assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                # Make sure that this is not fired twice
                self._grad_to_be_reduced[index] = False
                param.grad /= self.world_size

                # Future work includes clearing up the buffer if possible
                def cleanup() -> None:
                    if dst_rank != self.global_rank:
                        param.grad = None

                # Async reduce for this buffer, log the future
                optimizer.work_handles.append(
                    Workhandle(
                        handle=dist.reduce(
                            tensor=param.grad.data, dst=dst_rank, group=self.process_group, async_op=True
                        ),
                        callback=cleanup,
                    )
                )

                # If all the reduce operations have been called, add the gatekeeper
                if len(optimizer.work_handles) == optimizer._max_work_handles:
                    gatekeeper()

        # Bucket, update status, and possibly unroll the results
        def reduce_bucket(*_: Any) -> None:
            # Skip gradient reduction, do not alter status flags
            if not self.should_accumulate_grads and self._grad_to_be_reduced[index]:
                assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                # Make sure that this is not fired twice
                self._grad_to_be_reduced[index] = False

                # Copy to the flat buffer, update the buffer state
                bucket = optimizer.buckets[param.device][dst_rank]

                assert bucket.append(param, use_gradient=True), "Bucket overflow: max %s - current %s - adding %s" % (
                    bucket.max_size,
                    bucket.current_offset,
                    param.grad.numel(),
                )

                if bucket.full():

                    def unwrap() -> None:
                        for flat in bucket.params:
                            if dst_rank != self.global_rank:
                                # this rank is not the owner, release the grad
                                flat.param.grad = None
                            else:
                                # this rank is the owner, unroll the results
                                assert flat.param.grad is not None

                                flat.param.grad.data.copy_(
                                    bucket.buffer[flat.start : flat.stop].view_as(flat.param.data), non_blocking=True
                                )

                        bucket.reset()

                    bucket.buffer /= self.world_size

                    optimizer.work_handles.append(
                        Workhandle(
                            handle=dist.reduce(
                                tensor=bucket.buffer, dst=dst_rank, group=self.process_group, async_op=True,
                            ),
                            callback=unwrap,
                        )
                    )

                    # If all the reduce operations have been called, add the gatekeeper
                    if len(optimizer.work_handles) == optimizer._max_work_handles:
                        gatekeeper()

        return reduce_bucket if should_bucket else reduce_direct

    def _setup_backward_hooks(self) -> None:
        """
        Attach a reduce function to each grad-requiring parameter.
        This makes the gradient reduction automatic whenever there's a backward pass
        """

        # Go through the parameters, attach the hook
        for sharded_optimizer in self.sharded_optimizers:
            for param, should_bucket in sharded_optimizer.should_bucket_param.items():
                if param.grad is not None and param.grad.requires_grad:
                    raise RuntimeError("ShardedDataParallel only works with gradients that don't require grad")

                # Register the hook to the next function in line,
                # so that the hook is fired when this grad has properly been computed
                p_tmp = param.expand_as(param)
                assert p_tmp.grad_fn is not None
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                dst_rank = sharded_optimizer.param_to_rank[param]
                index = len(self._grad_accs)

                grad_acc.register_hook(self._get_reduce_fn(index, param, should_bucket, dst_rank, sharded_optimizer))
                self._grad_accs.append(grad_acc)  # keep this function in scope
