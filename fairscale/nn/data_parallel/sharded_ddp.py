# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A nn.Module wrapper to go with a Sharded Optimizer in order to handle targeted gradient
reduction automatically.
"""

from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union

import torch
from torch import nn
from torch.autograd import Variable
import torch.distributed as dist

from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
from fairscale.optim.utils import Workhandle


def _trainable(param: torch.Tensor) -> bool:
    return param.requires_grad


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
            The max size of the buffer used to batch the small parameter tensors, in number of elements (default 0 - unused).
            this will impact the long term memory consumption, because these buckets correspond to parameters which will not be sharded.
            Set to 0 to remove all bucketing, 1M to 8M is usually reasonable.
        auto_refresh_trainable (bool):
            (default: True) Check whether the parameters trainability (`requires_grad`) has changed and update both ShardedDDP
            and OSS automatically if this is the case. If set to False, `refresh_trainable()` needs to be called anytime
            a parameter is frozen or unfrozen.
        reduce_fp16 (bool):
            cast the grads to fp16 before reducing. Not needed if the model is already fp16, but will probably improve performance
            for multi node jobs using PyTorch AMP. The effect is similar to DDP's fp16_compress_hook_ and will also save some memory.

    .. _fp16_compress_hook: https://pytorch.org/docs/1.8.0/ddp_comm_hooks.html?highlight=fp16#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

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
        If `auto_refresh_trainable` is set to `True` (this is the default) then any trainability change in the model graph will be handled
        automatically.
        If `auto_refresh_trainable` is set to `False`, ShardedDDP will not refresh its assumptions with respect to trainable parameters
        for every forward pass, in the hope of saving some time. If some parameters are frozen or unfrozen over time, please refresh
        ShardedDDP assumptions by calling `refresh_trainable()` just after said change (before the next forward pass).

    """

    def __init__(
        self,
        module: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any = None,
        broadcast_buffers: bool = True,
        sync_models_at_startup: bool = True,
        reduce_buffer_size: int = 2 ** 23,
        auto_refresh_trainable: bool = True,
        reduce_fp16: bool = False,
    ):
        super().__init__()

        self.module = module
        self.sharded_optimizers = [sharded_optimizer] if not isinstance(sharded_optimizer, list) else sharded_optimizer
        self.enable_broadcast_buffers = broadcast_buffers
        self.auto_refresh_trainable = auto_refresh_trainable
        self.reduce_fp16 = reduce_fp16
        if reduce_buffer_size > 0 and reduce_fp16:
            self.reduce_fp16 = False
            logging.warning(
                "fp16 gradient reduction is not compatible with reduction buffers, which are requested. fp16 grad reduction is deactivated."
            )

        # Handle a no_sync() context which prevents the gradient synchronization,
        # accumulate in place
        self.should_accumulate_grads = False
        self.accumulate_grads_flipped = False

        # Communication related attributes
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.backend = dist.get_backend(self.process_group)
        self.world_size_scaling = 1.0 / dist.get_world_size(self.process_group)  # > 0
        self.reference_global_rank = OSS.get_global_rank(self.process_group, 0)  # picking rank 0 as the reference
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = OSS.get_global_rank(self.process_group, self.rank)
        self._local_to_global_rank = [
            OSS.get_global_rank(self.process_group, i) for i in range(dist.get_world_size(self.process_group))
        ]

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
        self._all_params = list(
            chain(
                *[sum([sum(p, []) for p in optim.per_device_params.values()], []) for optim in self.sharded_optimizers]
            )
        )
        self._trainable_params: List[torch.Tensor] = []
        self._grad_to_be_reduced: List[bool] = []
        self._trainable_param_to_rank: Dict[torch.Tensor, int] = {}
        self._reference_trainable_mask = list(map(_trainable, self._all_params))

        # - setup buckets and tensor views
        model_size = sum([p.numel() for p in self.module.parameters()])
        self.buffer_max_size = min(reduce_buffer_size, model_size)

        if dist.get_world_size(self.process_group) == 1:
            self.buffer_max_size = 0
            logging.info("Training is not really distributed, single rank. Deactivating buckets")

        logging.info(
            "ShardedDDP bucket size: {:.2f}M parameters, model size {:.2f}M parameters".format(
                self.buffer_max_size / 2 ** 20, model_size / 2 ** 20
            )
        )
        self.use_buckets = self.buffer_max_size > 0

        self.buckets: Dict[torch.device, Dict[int, GradBucket]] = {}
        self._should_bucket_grad: List[bool] = []
        self._bucket_list: List[GradBucket] = []

        # - setup backward hooks which will be called by Torch's autograd in due time
        self._grad_accs: List[Callable] = []
        self._grad_hooks: List[Any] = []
        self._manual_reduce: List[Callable] = []

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)

        # Make sure that all ranks start with the same model
        if sync_models_at_startup:
            self._sync_params_and_buffers()

        self._work_handles: Deque[Workhandle] = deque()
        self._bucket_flush_callback_set = False

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Module forward pass, handles any DDP-specific work in the background. Primes the
        backward pass for gradient reduction to the proper ranks.
        """

        # Deferred initialization, or change detection
        needs_setup = len(self._grad_hooks) == 0

        if self.auto_refresh_trainable:
            # Optionally check whether the trainable parameters have changed
            trainable_mask = list(map(_trainable, self._all_params))
            if trainable_mask != self._reference_trainable_mask:
                logging.warning("ShardedDDP detected that the trainable params changed, updating the partitioning")
                needs_setup = True
                self._reference_trainable_mask = trainable_mask

        if needs_setup:
            self.refresh_trainable()

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

        .. warning:
            Device changes are not supported, and this will raise an exception. The issue in that case is not
            really ShardedDDP, but OSS which will not be aware of the device change, and whose buffers will be
            in a broken state.

        Arguments:
            device (:class:`torch.device`): the desired device of the parameters and buffers in this module.
            dtype (:class:`torch.dtype`): the desired floating point type of the floating point parameters and buffers.
            non_blocking (bool): make it an asynchronous call.

        Returns:
            Module: self.
        """

        assert device in self.buckets.keys(), "Changing devices is not supported, because this would break OSSs state"
        assert (
            len(self.buckets.keys()) == 1
        ), "Several devices specified to begin with, incompatible with setting a single device here"

        for _device in self.buckets.keys():
            for bucket in self.buckets[_device].values():
                bucket.to(device=_device, dtype=dtype, non_blocking=non_blocking)

        self.module.to(device=device, dtype=dtype, non_blocking=non_blocking)

    def refresh_trainable(self) -> None:
        """ If the module trainability has changed, update all the assumptions """

        # Make sure that this is not done while gradients are waiting to be reduced (if no_sync context for instance)
        assert not functools.reduce(
            lambda x, y: x or y, self._grad_to_be_reduced, False
        ), "Grads waiting to be reduced: {}".format(self._grad_to_be_reduced)

        self._trainable_params = list(filter(lambda x: x.requires_grad, self._all_params))
        self._trainable_params.sort(key=lambda x: x.numel())
        self._grad_to_be_reduced = [True for _ in self._trainable_params]

        self._trainable_param_to_rank = {}
        for optim in self.sharded_optimizers:
            # OSS may need to change the communication pattern
            optim.refresh_trainable()

            # Update ShardedDDP given the new partitions
            for (
                device_per_rank_params
            ) in optim.per_device_params.values():  # all the params on this device (inc all ranks)
                for device_params in device_per_rank_params:
                    for param in filter(lambda x: x.requires_grad, device_params):
                        self._trainable_param_to_rank[param] = optim.param_to_rank[param]

        self._setup_bucket_strategy()
        self._setup_backward_hooks()

    def reduce(self) -> None:
        """
        This does not *need* to be called, the gradient reduction is done automatically during the BW pass.
        Use this method to reduce the gradients manually
        """

        # Check that this is not a mistake, if there's nothing to reduce
        assert functools.reduce(
            lambda x, y: x or y, self._grad_to_be_reduced, False
        ), "No grads waiting to be reduced, maybe that this was called twice or there was no BW pass ?"

        # Trigger all the current BW hooks
        self._bucket_flush_callback_set = True  # no need to flush in the end, we own the callback execution
        _ = list(map(lambda x: x(), self._manual_reduce))

        # Make sure that all the futures are consumed
        self._consume_work_handles()

    @torch.no_grad()
    def sync_buffers(self, blocking: bool = False) -> None:
        """
        Sync all the param buffers in between ranks (including for instance batch norm statistics).

        Arguments:
            blocking (bool): wait for the operation to conclude.
        """

        work_handles = []

        for buffer in self.module.buffers(recurse=True):
            work_handles.append(
                dist.broadcast(buffer.data, self.reference_global_rank, self.process_group, async_op=True)
            )

        if blocking and work_handles:
            if self.backend != dist.Backend.NCCL:
                _ = list(filter(lambda x: x.wait(), work_handles))
            else:
                work_handles[-1].wait()

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """

        for index, trainable_param in enumerate(self._all_params):
            if set_to_none and not self._should_bucket_grad[index]:
                trainable_param.grad = None
            elif trainable_param.grad is not None:
                trainable_param.grad.zero_()

        for bucket in self._bucket_list:
            bucket.zero()

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
        self.accumulate_grads_flipped = self.should_accumulate_grads != old_should_accumulate_grads
        self.should_accumulate_grads = old_should_accumulate_grads

    @torch.no_grad()
    def _clear_counters(self) -> None:
        """Reset all the grad reduce and call counters"""
        self._grad_to_be_reduced = [True for _ in self._grad_to_be_reduced]
        self._bucket_flush_callback_set = False

        if self.use_buckets:
            for bucket in self._bucket_list:
                bucket.reset_checked_in()

        if not self.should_accumulate_grads:
            self.accumulate_grads_flipped = False

    def _get_reduce_fn(self, index: int, param: torch.Tensor, dst_rank: int) -> Callable:
        """
        Two possible backward hooks for a given parameter: either directly reduce to the appropriate rank,
        or contribute to a bucket and reduce when the bucket is full.

        Either way a delayed action is necessary and is passed as a callback.
        """

        if not self.use_buckets or not self._should_bucket_grad[index]:
            # Direct reduction
            @torch.no_grad()
            def reduce(*_: Any) -> None:
                # Skip gradient reduction, do not alter status flags
                if not self.should_accumulate_grads and self._grad_to_be_reduced[index]:
                    assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                    if not self._bucket_flush_callback_set:
                        Variable._execution_engine.queue_callback(self._flush_reduce_calls)
                        self._bucket_flush_callback_set = True

                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False
                    param.grad.mul_(self.world_size_scaling)

                    if self.reduce_fp16:
                        param.grad.data = param.grad.data.half()

                    # Future work includes clearing up the buffer if possible
                    def cleanup() -> None:
                        if dst_rank != self.global_rank:
                            param.grad = None
                        else:
                            assert param.grad is not None
                            param.grad.data = param.grad.data.to(dtype=param.dtype)

                    # Async reduce for this buffer, log the future
                    self._work_handles.append(
                        Workhandle(
                            handle=dist.reduce(
                                tensor=param.grad.data,
                                dst=self._local_to_global_rank[dst_rank],
                                group=self.process_group,
                                async_op=True,
                            ),
                            callback=cleanup,
                        )
                    )

                    # Opportunistically try to empty the queue, free memory
                    self._try_consume_work_handle()

        else:

            @torch.no_grad()
            def reduce(*_: Any) -> None:
                # Skip gradient reduction, do not alter status flags

                if not self.should_accumulate_grads and self._grad_to_be_reduced[index]:
                    assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                    if not self._bucket_flush_callback_set:
                        Variable._execution_engine.queue_callback(self._flush_reduce_calls)
                        self._bucket_flush_callback_set = True

                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False
                    bucket = self.buckets[param.device][dst_rank]
                    bucket.params_checked_in += 1

                    if bucket.all_checked_in:
                        assert bucket.buffer is not None

                        # Normalize the bucket in one go
                        bucket.buffer.mul_(self.world_size_scaling)

                        # Reduce the bucket
                        bucket.sent = True
                        self._work_handles.append(
                            Workhandle(
                                handle=dist.reduce(
                                    tensor=bucket.buffer,
                                    dst=bucket.destination,
                                    group=self.process_group,
                                    async_op=True,
                                ),
                                callback=None,
                            )
                        )

                    # Opportunistically try to empty the queue
                    self._try_consume_work_handle()

        return reduce

    def _setup_backward_hooks(self) -> None:
        """
        Attach a reduce function to each grad-requiring parameter.
        This makes the gradient reduction automatic whenever there's a backward pass
        """

        # Detach possible pre-existing hooks
        while len(self._grad_hooks) > 0:
            self._grad_hooks.pop().remove()

        # Go through the parameters, attach the hook
        self._grad_accs = []
        self._manual_reduce = []
        for index, param in enumerate(self._trainable_params):
            if param.grad is not None and param.grad.requires_grad:
                raise RuntimeError("ShardedDataParallel only works with gradients that don't require grad")

            # Register the hook to the next function in line,
            # so that the hook is fired when this grad has properly been computed
            p_tmp = param.expand_as(param)
            assert p_tmp.grad_fn is not None
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            dst_rank = self._trainable_param_to_rank[param]

            reduce_function = self._get_reduce_fn(index, param, dst_rank)

            self._grad_hooks.append(grad_acc.register_hook(reduce_function))
            self._grad_accs.append(grad_acc)  # keep this hook in scope
            self._manual_reduce.append(reduce_function)

    @torch.no_grad()
    def _sync_params_and_buffers(self) -> None:
        """
        Sync the complete model states in between the ranks
        """

        work_handles = []

        for t in self.module.state_dict().values():
            work_handles.append(
                dist.broadcast(t, src=self.reference_global_rank, group=self.process_group, async_op=True)
            )

        # gloo does not guarantee inlining like NCCL, wait for all requests
        if self.backend != dist.Backend.NCCL:
            _ = list(filter(lambda x: x.wait(), work_handles))
        elif work_handles:
            work_handles[-1].wait()

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
        """Devise a bucketing strategy on a per-rank ownership level.
        These buckets will not be sharded, since the gradients would be re-allocated during the backward in that case.
        This method can be a slow for big models, but it it not typically called often (not for every forward for instance)
        """

        if not self.use_buckets:
            return

        # Devise the bucketing strategy. Parameters are already sorted, in that:
        # - these are only the trainable parameters, so they should produce grads
        # - they are sorted by increasing size
        self.buckets = {}
        self._should_bucket_grad = [False for _ in self._trainable_params]

        for i, param in enumerate(self._trainable_params):
            device = param.device
            dst_rank = self._trainable_param_to_rank[param]

            if param.device not in self.buckets.keys():
                self.buckets[param.device] = {}

            if dst_rank not in self.buckets[param.device].keys():
                self.buckets[param.device][dst_rank] = GradBucket(
                    self.buffer_max_size,
                    dtype=param.dtype,
                    device=param.device,
                    destination=self._local_to_global_rank[dst_rank],
                )

            # Criteria to decide whether this parameter is to be bucketed or not:
            # - enough room in the bucket
            if self.buckets[device][dst_rank].can_add_grad_view(param):
                self.buckets[device][dst_rank].add_grad(param)
                self._should_bucket_grad[i] = True

        self._bucket_list = list(chain(*[self.buckets[device].values() for device in self.buckets.keys()]))

        # Resize the buckets to remove lost space in the end
        for bucket in self._bucket_list:
            bucket.shrink()

    def _consume_work_handles(self) -> None:
        """Consume all the futures which are tied to this optimizer's buckets.
            We start from the first/older ones, since they are the most likely to be ready and non-blocking
            """

        while len(self._work_handles) > 0:
            work_handle = self._work_handles.popleft()
            work_handle.handle.wait()
            if work_handle.callback is not None:
                work_handle.callback()

    def _try_consume_work_handle(self) -> None:
        """Try to consume the oldest future. This is non blocking, if not ready we'll pass"""
        while len(self._work_handles) > 0 and self._work_handles[0].handle.is_completed():
            work_handle = self._work_handles.popleft()
            if work_handle.callback is not None:
                work_handle.callback()

    def _flush_reduce_calls(self) -> None:
        for bucket in self._bucket_list:
            if not bucket.sent:
                assert bucket.buffer is not None

                # Normalize the bucket in one go
                bucket.buffer.mul_(self.world_size_scaling)

                # Reduce the bucket
                self._work_handles.append(
                    Workhandle(
                        handle=dist.reduce(
                            tensor=bucket.buffer, dst=bucket.destination, group=self.process_group, async_op=True,
                        ),
                        callback=None,
                    )
                )
                bucket.sent = True

        self._consume_work_handles()
