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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim import OSS
from fairscale.optim.utils import Bucket, Workhandle


class ShardedDataParallel(nn.Module):
    """ Wrap the model, and reduce the gradients to the right rank during the backward pass.

    - the partition is given by the sharded optimizer
    - wrap the base model with a model which knows where to reduce each gradient
    - add an autograd function which calls the model grad dispatch on the way back

     Args:
        module (nn.Module):
            model to be wrapped
        sharded_optimizer (OSS, or list of OSS):
            the sharded optimizer(s) which will decide the gradient partitioning

    Keyword Args:
        process_group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffers (bool):
            Whether to additionally broadcast model buffers in between ranks at the beginning of each forward pass.
            Same setting as in Pytorch DDP, this is in addition to the broadcast and reduction of the model parameters.
        sync_models_at_startup (bool):
            Synchronize the models in between the ranks when starting up. Not needed if each rank has the same seed,
            or the training restarts from a saved state
        reduce_buffer_size (int):
            the max size of the buffer used to batch the small parameter tensors, in number of elements (default 8M).
            this will impact the long term memory consumption, because these buckets correspond to parameters which will not be sharded.
            Set to 0 to remove all bucketing.


    .. warning:
        ShardedDDP implements gradient sharding, meaning that each rank only owns a unique shard of the model gradients
        after the backward pass, in order to save memory and some communication bandwidth.

    .. warning:
        As a consequence of sharding:
            * in case of gradient clipping, one has to use the `clip_grad_norm` exposed by
                the `optimizer state sharding wrapper <fairscale.optim.OSS>`

            * after loss.backward() (or equivalent) each rank will have `None` in place of some param.grad

            * Pytorch and Apex AMP implementations will hang when used in conjunction with `ShardedDDP`.
                One needs a `shard-aware grad scaler<ShardedGradScaler>`, which is proposed in `fairscale.optim.grad_scaler`,
                compatible with PytorchAMP.

    .. warning:
        ShardedDDP uses buckets to speed up the network communications. If some parameters require_grad but are not actually
        used, there is a chance that this would prevent the bucket mechanism to function, and that this could not be automatically
        handled. In that case ShardedDDP will raise an exception and suggest to either remove the unused parameters from your model
        (https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=unused_parameters is helpful)
        or set `reduce_buffer_size` to 0
    """

    def __init__(
        self,
        module: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any = None,
        broadcast_buffers: bool = True,
        sync_models_at_startup: bool = True,
        reduce_buffer_size: int = 2 ** 23,
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
        self.world_size_scaling = 1.0 / dist.get_world_size(self.process_group)  # > 0
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
        # several optimizers can be present each working on seperate parameter set which is spread across multiple ranks

        # - we build an iterator which goes through all the parameters involved globally
        all_param_iterator = chain(
            *[sum([sum(p, []) for p in optim.per_device_params.values()], []) for optim in self.sharded_optimizers]
        )
        self._grad_to_be_reduced = [True for _ in filter(lambda x: x.requires_grad, all_param_iterator)]

        # - keep track of the grads which have already been reduced
        self._reduced_grads: Dict[OSS, int] = {}
        self._reduced_grads_max = {o: len(o.param_to_rank.values()) for o in self.sharded_optimizers}

        # - setup buckets and tensor views
        model_size = sum([p.numel() for p in self.module.parameters()])
        if dist.get_world_size(self.process_group) <= 8:
            logging.info("Assuming single node environment. De-activating ShardedDDP buckets")
            reduce_buffer_size = 0

        self.buffer_max_size = min(reduce_buffer_size, model_size)
        logging.info(
            "ShardedDDP bucket size: {:.2f}M parameters, model size {:.2f}M parameters".format(
                self.buffer_max_size / 2 ** 20, model_size / 2 ** 20
            )
        )
        self.use_buckets = self.buffer_max_size > 0

        self.buckets: Dict[OSS, Dict[torch.device, List[Bucket]]] = {o: {} for o in self.sharded_optimizers}
        self._should_bucket_grad: List[bool] = []
        self._setup_bucket_strategy()

        # - setup backward hooks which will be called by Torch's autograd in due time
        self._grad_accs: List[Callable] = []
        self._setup_backward_hooks()

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)

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

        # Reset all the grad reduce and bucket state flags
        self._clear_counters()

        # Normal FW on the base model
        return self.module(*inputs, **kwargs)

    def to(  # type: ignore
        self,
        device: Optional[Union[int, torch.device]],
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "ShardedDataParallel":
        """
        Moves and/or casts the parameters and buffers.

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (:class:`torch.device`): the desired device of the parameters and buffers in this module.
            dtype (:class:`torch.dtype`): the desired floating point type of the floating point parameters and buffers.
            non_blocking (bool): make it an asynchronous call.

        Returns:
            Module: self.

        """

        for optimizer in self.buckets.keys():
            for device in self.buckets[optimizer].keys():
                for bucket in self.buckets[optimizer][device]:
                    bucket.buffer.to(device=device, dtype=dtype, non_blocking=non_blocking)

        self.module.to(device)

    def reduce(self) -> None:
        """.. deprecated:: 0.0.4

        This does not need to be called, the gradient reduction is done automatically during the BW pass
        """
        logging.warning("This is not useful anymore, gradients have been reduced automatically with the backward pass")

    @torch.no_grad()
    def sync_buffers(self, blocking: bool = False) -> None:
        """
        Sync all the param buffers in between ranks (including for instance batch norm statistics).
        """

        last_work_handle = None

        for buffer in self.module.buffers(recurse=True):
            last_work_handle = dist.broadcast(
                buffer.data, self.reference_global_rank, self.process_group, async_op=True
            )

        if blocking and last_work_handle:
            # Only wait for the last coms, they're inlined on the same CUDA stream
            last_work_handle.wait()

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    @contextlib.contextmanager
    def no_sync(self) -> Generator:
        """A context manager to disable gradient synchronization."""
        old_should_accumulate_grads = self.should_accumulate_grads
        self.should_accumulate_grads = True
        yield
        self.should_accumulate_grads = old_should_accumulate_grads

    @torch.no_grad()
    def _clear_counters(self) -> None:
        """Reset all the grad reduce and call counters"""
        if not self.should_accumulate_grads:

            self._grad_to_be_reduced = [True for _ in self._grad_to_be_reduced]
            self._reduced_grads = {o: 0 for o in self.sharded_optimizers}

            for optimizer in self.buckets.keys():
                for device in self.buckets[optimizer].keys():
                    for bucket in self.buckets[optimizer][device]:
                        assert bucket.sent, (
                            "A bucket failed to be sent, probably unused parameters."
                            + "Either remove the unused parameter or de-activate ShardedDDP buckets -set reduce_buffer_size to 0-"
                        )

                        bucket.reset()

    def _find_rank(self, param: Parameter) -> Tuple[OSS, int]:
        """ Look up where this parameter belongs to """
        for optim in self.sharded_optimizers:
            if param in optim.param_to_rank.keys():
                return optim, optim.param_to_rank[param]

        assert False, "This parameter is not present in an optimizer, this should not happen"
        return (None, -1)

    def _get_reduce_fn(self, index: int, param: torch.Tensor, dst_rank: int, optimizer: OSS) -> Callable:
        """
        Two possible backward hooks for a given parameter: either directly reduce to the appropriate rank,
        or contribute to a bucket and reduce when the bucket is full.

        Either way a delayed action is necessary and is passed as a callback.
        """

        @torch.no_grad()
        def reduce(*_: Any) -> None:
            # Skip gradient reduction, do not alter status flags
            if not self.should_accumulate_grads and self._grad_to_be_reduced[index]:
                assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                # Make sure that this is not fired twice
                self._grad_to_be_reduced[index] = False

                if not self.use_buckets or not self._should_bucket_grad[index]:
                    param.grad.mul_(self.world_size_scaling)

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
                    self._reduced_grads[optimizer] += 1
                else:
                    bucket = self.buckets[optimizer][param.device][dst_rank]
                    bucket.params_checked_in += 1

                    if bucket.full():
                        # Normalize the bucket in one go
                        bucket.buffer.mul_(self.world_size_scaling)

                        # Reduce the bucket
                        bucket.sent = True
                        optimizer.work_handles.append(
                            Workhandle(
                                handle=dist.reduce(
                                    tensor=bucket.buffer, dst=dst_rank, group=self.process_group, async_op=True,
                                ),
                                callback=None,
                            )
                        )
                        self._reduced_grads[optimizer] += 1

                # Opportunistically try to empty the queue
                optimizer._try_consume_work_handle()

                # If all the reduce operations have been called,
                # make sure that all the asynchronous calls have concluded before moving on
                # and execute the delayed actions (release gradients, unroll the buckets)
                if self._reduced_grads[optimizer] == self._reduced_grads_max[optimizer]:
                    optimizer._consume_work_handles()

        return reduce

    def _setup_backward_hooks(self) -> None:
        """
        Attach a reduce function to each grad-requiring parameter.
        This makes the gradient reduction automatic whenever there's a backward pass
        """

        # Go through the parameters, attach the hook
        for sharded_optimizer in self.sharded_optimizers:
            for (
                device_per_rank_params
            ) in sharded_optimizer.per_device_params.values():  # all the params on this device (inc all ranks)
                for device_params in device_per_rank_params:
                    for param in filter(lambda x: x.requires_grad, device_params):
                        if param.grad is not None and param.grad.requires_grad:
                            raise RuntimeError("ShardedDataParallel only works with gradients that don't require grad")

                        # Register the hook to the next function in line,
                        # so that the hook is fired when this grad has properly been computed
                        p_tmp = param.expand_as(param)
                        assert p_tmp.grad_fn is not None
                        grad_acc = p_tmp.grad_fn.next_functions[0][0]
                        dst_rank = sharded_optimizer.param_to_rank[param]
                        index = len(self._grad_accs)

                        grad_acc.register_hook(self._get_reduce_fn(index, param, dst_rank, sharded_optimizer))
                        self._grad_accs.append(grad_acc)  # keep this function in scope

        #  Add a hook on the module to flush the buckets, if needed
        if self.use_buckets:

            def bucket_flush(*unused: Any) -> None:
                handle = None

                for bucket_optim in self.buckets.values():
                    for bucket_rank in bucket_optim.values():
                        for bucket in bucket_rank:
                            if not bucket.sent:
                                # Reduce the bucket. Some parameters went unused and this bucket was not flushed
                                bucket.buffer.mul_(self.world_size_scaling)
                                bucket.sent = True
                                handle = dist.reduce(
                                    tensor=bucket.buffer,
                                    dst=bucket.destination,
                                    group=self.process_group,
                                    async_op=True,
                                )

                # Only wait on the last handle
                if handle:
                    handle.wait()

            self.module.register_backward_hook(bucket_flush)

    @torch.no_grad()
    def _sync_params_and_buffers(self) -> None:
        """
        Sync the complete model states in between the ranks
        """

        last_work_handle = None

        for t in self.module.state_dict().values():
            last_work_handle = dist.broadcast(
                t, src=self.reference_global_rank, group=self.process_group, async_op=True
            )

        # Only wait for the last handle, they're inlined in the same CUDA stream
        if last_work_handle:
            last_work_handle.wait()

    def _passing_sync_batchnorm_handle(self, module: nn.Module) -> None:
        """
        Passes handle required for ``torch.nn.modules.SyncBatchNorm``.
        Adapted from ``torch.nn.distributed.DistributedDataParallel``.
        """
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                assert self.device_type != "cpu", "SyncBatchNorm layers only work with GPU modules"
                # device_id logic has not been handled, assume single-process single-device
                # SyncBatchNorm only supports DDP with single-process single-device anyway'
                layer._specify_ddp_gpu_num(1)  # type: ignore

    def _setup_bucket_strategy(self) -> None:
        """Devise a bucketing strategy on a per-rank ownership level. These buckets will not be sharded, since the gradients would be re-allocated during the backward in that case.
        """

        if not self.use_buckets:
            return

        # - Allocate one buffer per rank and per device to group the small parameters
        for sharded_optimizer in self.sharded_optimizers:
            for device, per_device in sharded_optimizer.per_device_params.items():
                self.buckets[sharded_optimizer][device] = [
                    Bucket(buffer=torch.zeros(self.buffer_max_size, dtype=per_device[0][0].dtype, device=device))
                    for _ in per_device
                ]

        # Devise the bucketing strategy
        for sharded_optimizer in self.sharded_optimizers:
            for device, per_rank_params in sharded_optimizer.per_device_params.items():
                for dst_rank, params in enumerate(per_rank_params):
                    offset = 0
                    bucket = self.buckets[sharded_optimizer][device][dst_rank]
                    bucket.destination = dst_rank

                    for param in filter(lambda x: x.requires_grad is True, params):
                        # Criteria to decide whether this parameter is to be bucketed or not:
                        # - enough room in the bucket
                        if (offset + param.numel()) < self.buffer_max_size:
                            self._should_bucket_grad.append(True)

                            # This parameter gradients becomes a view of the bucket
                            offset_next = offset + param.numel()

                            if param.grad is None:
                                # will be overwritten just below, see next line
                                param.grad = torch.zeros_like(param)

                            param.grad.data = bucket.buffer[offset:offset_next].view_as(param.data)
                            offset = offset_next

                            # Update the bucket
                            self._reduced_grads_max[sharded_optimizer] -= 1  # one less reduce call per bucketed grad
                            self.buckets[sharded_optimizer][device][dst_rank].max_params_checked_in += 1

                        else:
                            self._should_bucket_grad.append(False)

                    # Resize the bucket to remove lost space in the end
                    bucket.buffer.resize_(offset)
                    if bucket.max_params_checked_in > 0:
                        self._reduced_grads_max[sharded_optimizer] += 1  # one reduce call per bucket
