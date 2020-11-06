# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A module wrapper to go with a Sharded Optimizer in order to handle targeted gradient reduction/gathering automatically.
"""

import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim.oss import OSS


def _get_global_rank(group: Any, rank: int) -> int:
    return rank if group is dist.group.WORLD else dist.distributed_c10d._get_global_rank(group, rank)  # type: ignore


class Gatekeeper(torch.autograd.Function):
    """
     The gatekeeper layer makes sure that the reduce is done before the optimizer steps in
     - In the forward pass it does nothing
     - In the backward pass, it gathers gradients to the owner.
     NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
     """

    @staticmethod
    def forward(ctx: Any, work_queue: List[Any], *inputs: Any) -> Any:  # type: ignore
        # Store a handle to the work queue for the BW reduce
        ctx.work_queue = work_queue
        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        # Consume the handles, make sure that all the reduces are done before the optimizer can step
        while len(ctx.work_queue) > 0:
            wh, callback = ctx.work_queue.pop()
            if wh is not None:
                wh.wait()
                callback()

        return tuple([None, *grad_outputs])


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
        reduce_buffer_size: int = 2 ** 19,
    ):
        super().__init__()

        self.base_model = base_model
        self.sharded_optimizers = [sharded_optimizer] if isinstance(sharded_optimizer, OSS) else sharded_optimizer
        self.broadcast_buffers = broadcast_buffers

        # Communication related attributes
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.reference_global_rank = _get_global_rank(self.process_group, 0)  # picking rank 0 as the reference
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = _get_global_rank(self.process_group, self.rank)

        # Expose the same attributes as PytorchDDP, some frameworks rely on them.
        # See https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel
        self.is_multi_device_module = len({p.device for p in self.base_model.parameters()}) > 1
        distinct_device_types = {p.device.type for p in self.base_model.parameters()}
        assert len(distinct_device_types) == 1, (
            "ShardedDataParallel's input module must be on "
            "the same type of devices, but input module parameters locate in {}."
        ).format(distinct_device_types)
        self.device_type = list(distinct_device_types)[0]

        # Fill in a look-up table per grad
        self._grad_to_rank = [
            _get_global_rank(self.process_group, self._find_rank(p)[1])
            for p in filter(lambda x: x.requires_grad, self.base_model.parameters())
        ]

        # Setup the reduce buckets
        # reduce_buffer dimensions: [optimizer, device, rank params]
        # bucket_strategy dimensions: [optimizer, device, rank tuple -should bucket, offset_start, offset_end-]
        # bucket_state dimensions: [optimizer, device, rank - current offset, max offset-]
        self._reduce_buckets: Dict[OSS, Dict[torch.device, List[Dict[str, Any]]]] = {}
        self._bucket_strategy: List[Tuple[bool, int, int]] = []
        self._bucket_state: Dict[OSS, Dict[torch.device, List[Tuple[int, int]]]] = {}
        self._buffer_size = min(reduce_buffer_size, sum(p.numel() for p in self.base_model.parameters()))
        self._setup_buckets(reduce_buffer_size)

        # Scafolding to be able to reduce the grads during the BW pass
        self._grad_to_be_reduced = [True for _ in filter(lambda x: x.requires_grad, self.base_model.parameters())]
        self._grad_accs: List[Callable] = []
        self._reduce_work_handles: List[Any] = []
        self._setup_backward_hooks()

        # Make sure that all ranks start with the same model
        self.sync_all_params()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if self.broadcast_buffers:
            self.sync_buffers()

        # Reset all the grad reduce and bucket state flags
        self._grad_to_be_reduced = [True for _ in self._grad_to_be_reduced]
        for sharded_optimizer in self.sharded_optimizers:
            for device, per_rank_params in sharded_optimizer.per_device_params.items():
                for r in range(self.world_size):
                    self._bucket_state[sharded_optimizer][device][r] = (
                        0,
                        self._bucket_state[sharded_optimizer][device][r][1],
                    )

        # mark tensors as "requires_grad" to register the Gatekeeper in the autograd graph
        inputs = ShardedDataParallel._mark_inputs_requires_grad(*inputs)

        # Register the gatekeeper, reduce wise
        inputs = Gatekeeper.apply(self._reduce_work_handles, *inputs)

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
        work_handles = [
            dist.broadcast(t, src=self.reference_global_rank, group=self.process_group, async_op=True)
            for t in self.base_model.state_dict().values()
        ]

        _ = list(map(lambda x: x.wait(), work_handles))

    def sync_buffers(self, non_blocking: bool = False) -> None:
        """
        Sync all the param buffers in between ranks.
        """
        for x in self.base_model.buffers(recurse=True):
            dist.broadcast(x.data, self.reference_global_rank, self.process_group, async_op=True)

    @staticmethod
    def _mark_inputs_requires_grad(*inputs: Any) -> Any:
        if isinstance(inputs, torch.Tensor):
            inputs.requires_grad = True

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            for i in filter(lambda x: isinstance(x, torch.Tensor), inputs):
                i.requires_grad = True

        return inputs

    def _find_rank(self, param: Parameter) -> Tuple[OSS, int]:
        """ Look up where this parameter belongs to """
        for optim in self.sharded_optimizers:
            if param in optim.param_to_rank.keys():
                return optim, optim.param_to_rank[param]

        assert False, "This parameter is not present in an optimizer, this should not happen"

    def _setup_backward_hooks(self) -> None:
        """
        Attach a reduce function to each grad-requiring parameter. This makes the gradient reduction automatic whenever there's a BW pass
        """

        parameters_with_grad = list(filter(lambda x: x.requires_grad, self.base_model.parameters()))

        # Build one hook per parameter
        def get_reduce_fn(index: int) -> Callable:
            # Find the corresponding bucket strategy
            param = parameters_with_grad[index]
            optimizer, rank = self._find_rank(param)
            should_bucket, offset_start, offset_end = self._bucket_strategy[index]

            # Return the appropriate hook
            if not should_bucket:
                # Direct reduce
                def reduce(*unused: Any) -> None:
                    if param.grad is not None and self._grad_to_be_reduced[index]:
                        # Make sure that this is not fired twice
                        self._grad_to_be_reduced[index] = False

                        param.grad /= self.world_size

                        # Future work includes clearing up the buffer if possible
                        def cleanup() -> None:
                            if self._grad_to_rank[index] != self.global_rank:
                                param.grad = None

                        # Async reduce for this buffer, log the future
                        self._reduce_work_handles.append(
                            (
                                dist.reduce(
                                    param.grad.data, self._grad_to_rank[index], group=self.process_group, async_op=True,
                                ),
                                cleanup,
                            )
                        )

                return reduce

            else:
                # Bucket, update status, and possibly unroll the results
                def reduce(*unused: Any) -> None:
                    bucket = self._reduce_buckets[optimizer][param.device][rank]
                    current_fill, max_fill = self._bucket_state[optimizer][param.device][rank]

                    if param.grad is not None and self._grad_to_be_reduced[index]:
                        # Make sure that this is not fired twice
                        self._grad_to_be_reduced[index] = False

                        # Copy to the flat buffer, update the buffer state
                        bucket["buffer"][offset_start:offset_end].copy_(param.grad.data.view(-1))
                        current_fill += offset_end - offset_start
                        self._bucket_state[optimizer][param.device][rank] = (current_fill, max_fill)

                        # Update buffer state, and reduce if full
                        if current_fill == max_fill:

                            def unwrap() -> None:
                                for p, offset, end in bucket["params"]:
                                    if self._grad_to_rank[index] != self.global_rank:
                                        # this rank is not the owner, release the grad
                                        p.grad = None
                                    else:
                                        # this rank is the owner, unroll the results
                                        p.grad.data.copy_(bucket["buffer"][offset:end].view_as(p.data))

                            bucket["buffer"] /= self.world_size

                            self._reduce_work_handles.append(
                                (
                                    dist.reduce(
                                        bucket["buffer"],
                                        self._grad_to_rank[index],
                                        group=self.process_group,
                                        async_op=True,
                                    ),
                                    unwrap,
                                )
                            )

                return reduce

        # Go through the parameters, attach the hook
        for i, p in enumerate(filter(lambda x: x.requires_grad, self.base_model.parameters())):
            if p.grad is not None and p.grad.requires_grad:
                raise RuntimeError("ShardedDataParallel only works " "with gradients that don't require grad")

            # Register the hook to the next function in line, so that the hook is fired when this grad
            # has properly been computed
            p_tmp = p.expand_as(p)
            if p_tmp.grad_fn is not None:
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(get_reduce_fn(i))
                self._grad_accs.append(grad_acc)  # keep this function in scope

    def _setup_buckets(self, reduce_buffer_size: int) -> None:
        # Allocate reduce buffers
        # - One buffer per rank per device for each optimizer
        for sharded_optimizer in self.sharded_optimizers:
            self._reduce_buckets[sharded_optimizer] = {}
            for device, per_device in sharded_optimizer.per_device_params.items():
                self._reduce_buckets[sharded_optimizer][device] = [
                    {"buffer": torch.zeros(self._buffer_size, dtype=params[0].dtype, device=device)}
                    for params in per_device
                ]

        # Tag parameters to either bucket them or reduce them directly
        # -> For all params, save a 3-tuple: (bucket ?, bucket_start, bucket_end)
        # use the fact that the parameters are sorted to begin with, when the partition is queried
        # from the sharded optimizer
        parameters_with_grad = list(filter(lambda x: x.requires_grad, self.base_model.parameters()))
        self._bucket_strategy = [(False, -1, -1) for _ in parameters_with_grad]

        def find_param_index(p: Parameter) -> int:
            # small helper, needed because directly using .index() on a tensor list does not check for identity
            for i, pg in enumerate(parameters_with_grad):
                if pg is p:
                    return i

            assert False

        for sharded_optimizer in self.sharded_optimizers:
            self._bucket_state[sharded_optimizer] = {}

            for device, per_rank_params in sharded_optimizer.per_device_params.items():
                self._bucket_state[sharded_optimizer][device] = []
                for dst_rank, params in enumerate(per_rank_params):
                    offset = 0
                    self._reduce_buckets[sharded_optimizer][device][dst_rank]["params"] = []

                    for p in params:
                        index = find_param_index(p)
                        if offset + p.numel() < self._buffer_size:
                            # This parameter is small enough to fit in the remaining size of the bucket
                            end = offset + p.numel()
                            self._bucket_strategy[index] = (True, offset, end)
                            self._reduce_buckets[sharded_optimizer][device][dst_rank]["params"].append((p, offset, end))
                            offset = end
                        else:
                            # The parameters are sorted by size, so all the following parameters
                            # will be too big and can be skipped
                            break

                    # Register the max offset for this buffer
                    self._bucket_state[sharded_optimizer][device].append((0, offset))
