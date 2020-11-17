# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A nn.Module wrapper to go with a Sharded Optimizer in order to handle targeted gradient
reduction automatically.
"""

from contextlib import contextmanager
from itertools import chain
import logging
from typing import Any, Callable, Generator, List, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim import OSS


class ShardedDataParallel(nn.Module):
    """
    Wrap the model, and reduce the gradients to the right rank during the backward pass.

    - the partition is given by the sharded optimizer
    - wrap the base model with a model which knows where to reduce each gradient
    - add an autograd function which calls the model grad dispatch on the way back

     Args:
        base_model (nn.Module):
            model to be wrapped
        sharded_optimizer (OSS, or list of OSS):
            the sharded optimizer(s) which will decide the gradient partitioning
    Keyword Args:
        process_group (torch.nn.Optimizer):
            optimizer to shard (default: SGD)
        process_group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffers (bool):
            whether to broadcast model buffers in between ranks at the beginning of each forward pass
        reduce_buffer_size (int):
            the size of the per-device-per-rank buffers used for the reduce operation
    """

    def __init__(
        self,
        base_model: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any = None,
        broadcast_buffers: bool = True,
    ):
        super().__init__()

        self.base_model = base_model
        self.sharded_optimizers = [sharded_optimizer] if isinstance(sharded_optimizer, OSS) else sharded_optimizer
        self.broadcast_buffers = broadcast_buffers
        self.accumulate_grads = False

        # Communication related attributes
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.reference_global_rank = OSS.get_global_rank(self.process_group, 0)  # picking rank 0 as the reference
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = OSS.get_global_rank(self.process_group, self.rank)

        # Expose the same attributes as PytorchDDP, some frameworks rely on them.
        # See https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel
        self.is_multi_device_module = len({p.device for p in self.base_model.parameters()}) > 1
        distinct_device_types = {p.device.type for p in self.base_model.parameters()}
        assert len(distinct_device_types) == 1, (
            "ShardedDataParallel's input module must be on "
            "the same type of devices, but input module parameters are located on {} different device types."
        ).format(distinct_device_types)
        self.device_type = list(distinct_device_types)[0]

        # Scafolding to be able to reduce the grads during the BW pass
        self._param_iterator = chain(*[optim.param_bucket_strategy.keys() for optim in self.sharded_optimizers])
        self._grad_to_be_reduced = [True for _ in self._param_iterator]
        print(f"{len(self._grad_to_be_reduced)} grads to be reduced")
        self._grad_accs: List[Callable] = []
        self._setup_backward_hooks()

        # Make sure that all ranks start with the same model
        self.sync_all_params()

    def forward(self, *inputs: Any, **_: Any) -> Any:
        """
        Module forward pass, handles any DDP-specific work in the background. Primes the
        backward pass for gradient reduction to the proper ranks.
        """
        if self.broadcast_buffers:
            self.sync_buffers()

        # Reset all the grad reduce and bucket state flags
        self._grad_to_be_reduced = [True] * len(self._grad_to_be_reduced)

        # Normal FW on the base model
        return self.base_model(*inputs)

    def reduce(self) -> None:
        """ .. deprecated:: 0.0.4
            This does not need to be called, the gradient reduction is done automatically during the BW pass
        """
        logging.warning("This is not useful anymore, gradients have been reduced automatically with the backward pass")

    def sync_all_params(self) -> None:
        """
        Sync the complete model states in between the ranks
        """
        with torch.no_grad():
            work_handles = [
                dist.broadcast(t, src=self.reference_global_rank, group=self.process_group, async_op=True)
                for t in self.base_model.state_dict().values()
            ]

        _ = list(map(lambda x: x.wait(), work_handles))

    def sync_buffers(self) -> None:
        """
        Sync all the param buffers in between ranks.
        """
        with torch.no_grad():
            for buffer in self.base_model.buffers(recurse=True):
                dist.broadcast(buffer.data, self.reference_global_rank, self.process_group, async_op=True)

    @contextmanager
    def no_sync(self) -> Generator:
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

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
        # Return the appropriate hook
        def reduce_direct(*_: Any) -> None:
            # Skip gradient reduction, do not alter status flags
            if not self.accumulate_grads:
                if param.grad is not None and self._grad_to_be_reduced[index]:
                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False
                    param.grad /= self.world_size

                    # Future work includes clearing up the buffer if possible
                    def cleanup() -> None:
                        if dst_rank != self.global_rank:
                            param.grad = None

                    # Async reduce for this buffer, log the future
                    optimizer.work_handles.append(
                        (
                            dist.reduce(tensor=param.grad.data, dst=dst_rank, group=self.process_group, async_op=True),
                            cleanup,
                        )
                    )

                    # If all the reduce operations have been called, add the gatekeeper
                    if len(optimizer.work_handles) == optimizer._max_work_handles:
                        optimizer._consume_work_handles()

        # Bucket, update status, and possibly unroll the results
        def reduce_bucket(*_: Any) -> None:
            # Skip gradient reduction, do not alter status flags
            if not self.accumulate_grads:
                bucket = optimizer.buckets[param.device][dst_rank]

                if param.grad is not None and self._grad_to_be_reduced[index]:
                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False

                    # Copy to the flat buffer, update the buffer state
                    assert bucket.append(param, use_gradient=True)

                    if bucket.full():

                        def unwrap() -> None:
                            for param, offset, end in bucket.params:
                                if dst_rank != self.global_rank:
                                    # this rank is not the owner, release the grad
                                    param.grad = None
                                else:
                                    # this rank is the owner, unroll the results
                                    assert param.grad is not None

                                    param.grad.data.copy_(bucket.buffer[offset:end].view_as(param.data))

                            bucket.reset()

                        bucket.buffer /= self.world_size

                        optimizer.work_handles.append(
                            (
                                dist.reduce(
                                    tensor=bucket.buffer, dst=dst_rank, group=self.process_group, async_op=True,
                                ),
                                unwrap,
                            )
                        )

                        # If all the reduce operations have been called, add the gatekeeper
                        if len(optimizer.work_handles) == optimizer._max_work_handles:
                            optimizer._consume_work_handles()

        return reduce_bucket if should_bucket else reduce_direct

    def _setup_backward_hooks(self) -> None:
        """
        Attach a reduce function to each grad-requiring parameter.
        This makes the gradient reduction automatic whenever there's a backward pass
        """

        # Go through the parameters, attach the hook
        for sharded_optimizer in self.sharded_optimizers:
            for param, should_bucket in sharded_optimizer.param_bucket_strategy.items():
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
