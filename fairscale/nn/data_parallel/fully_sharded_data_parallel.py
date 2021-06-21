# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import time
import traceback
import typing
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Mapping, NamedTuple, Optional, Set, Tuple, Union

import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairscale.nn.misc import FlattenParamsWrapper
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from fairscale.utils.containers import apply_to_tensors
from fairscale.utils.parallel import (
    chunk_and_pad,
    enable_pytorch_sync_bn,
    get_process_group_cached,
    validate_process_group,
)
from fairscale.utils.params import broadcast_object, calc_grad_norm, recursive_copy_to_device
from fairscale.utils.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.utils.state_dict import replace_by_prefix_

from . import fsdp_optim_utils as ou

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class TrainingState(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.

    ..note::

        BACKWARD_PRE and BACKWARD_POST states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).

    TODO (Min): It would be nice to capture the stepping state as well.
        Maybe we can use the model.zero_grad() call, but not sure if it
        is called if optim.zero_grad() is used instead.
        It would be nice to have clear state transition be explicit like:

        zero_grad -> fwd -> bwd -> optionally accum grad by repeating
        fwd/bwd -> stepping -> loop back to zero_grad
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class FullyShardedDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters across data parallel workers. This
    is inspired by `Xu et al.`_ as well as the ZeRO Stage 3 from DeepSpeed_.

    .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
    .. _DeepSpeed: https://www.deepspeed.ai/

    Usage::

        import torch
        from fairscale.nn.data_parallel import FullyShardedDataParallel
        torch.cuda.set_device(device_id)
        sharded_module = FullyShardedDataParallel(my_module)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        x = sharded_module(x, y=3, z=torch.Tensor([1]))
        loss = x.sum()
        loss.backward()
        optim.step()

    It is also possible to shard individual layers separately and have an outer
    wrapper handle any leftover parameters. This can be helpful to further
    reduce GPU memory usage, reduce system memory usage when initializing large
    models and to improve training speed by overlapping the all-gather step
    across the forward pass. For example::

        import torch
        from fairscale.nn.auto_wrap import enable_wrap, auto_wrap
        from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
        fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=True, flatten_parameters=True)
        with enable_wrap(**fsdp_params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))
            assert isinstance(self.l1, FSDP)
            # Separately Wraps children modules with more than 1e8 params
            large_tfmr = torch.nn.Transformer(d_model=2048, encoder_layers=12, decoder_layers=12)
            self.l2 = auto_wrap(large_tfmr, min_num_params=1e8)
            assert isinstance(self.l2, FSDP)

    .. warning::

        The optimizer must be initialized *after* the module has been wrapped,
        since FSDP will shard parameters in-place and this will break any
        previously initialized optimizers.

    .. warning::

        If you wrap every parameter inside a nested FSDP and leaving the outer
        FSDP empty without any parameter, checkpointing activation may trigger
        an assert on the backward pass. The solution is to leave some parameters
        to the outer FSDP.

    .. warning::

        If activation checkpointing is used with FSDP, it is strongly encouraged
        to use ``checkpoint_wrapper`` function from FairScale instead of the
        ``checkpoint`` function from PyTorch.

    Args:
        module (nn.Module):
            module to be wrapped with FullyShardedDataParallel.
        process_group (Optional):
            process group for sharding
        reshard_after_forward (bool, Optional):
            if ``True``, reshard parameters after the forward pass. This saves
            memory but slows training. This is only relevant when resharding
            individual layers.
        mixed_precision (bool, Optional):
            if ``True``, inputs, activations and gradients will be kept in FP16;
            computation and communication will occur in FP16; and a (sharded)
            master copy of the model weights will be maintained in FP32.
        fp32_reduce_scatter (bool, Optional):
            if ``True``, then reduce-scatter gradients in FP32. This is only
            relevant when *``mixed_precision``* is ``True``.
        flatten_parameters (bool, Optional):
            if ``True``, flatten parameters into a single contiguous tensor,
            which improves training speed.
        move_params_to_cpu (bool, Optional):
            if ``True``, offload FP32 params to CPU. This is only relevant when
            *``mixed_precision``* is ``True``.
        compute_dtype (torch.dtype, Optional):
            dtype for full parameters for computation. This defaults to
            ``torch.float32`` unless *``mixed_precision``* is set, in which case
            it defaults to ``torch.float16``.
        buffer_dtype (torch.dtype, Optional):
            dtype for buffers for computation. This defaults to ``compute_dtype``.
        move_grads_to_cpu (bool, Optional):
            move gradient shard to CPU after reduction. This is useful when
            combined with CPU-based optimizers. It defaults to the value of
            *``cpu_offload``*.
        bucket_cap_mb (int, Optional):
            FSDP will bucket parameters so that gradient reduction can
            be more efficient for small parameters.
            ``bucket_cap_mb`` controls the bucket size in MegaBytes (MB). Buckets
            are sub-divided based on world_size, so the max shard size is roughly
            ``bucket_cap_mb / world_size``. There is one bucketer (with potentially
            multiple ``bucket_cap_mb`` sized buffers shared by all FSDP instances.
            Large gradient tensors are directly reduced without using the buffers.
            The buffers are there to reduce communication overhead for small tensors.
            Overlapping with computation happens due to use of a different CUDA stream
            than the computation CUDA stream. The total memory overhead per buffer is around
            ``bucket_cap_mb / world_size * (world_size + 1)``.
            The buffers are allocated during the backward pass and freed at the end
            of the backward pass to save more memory for other phases of the
            training process.
            Note, the memory vs. speed tradeoff of bucket size is very different
            from that of the DDP engine. In DDP, the buffer size ``1MB + n*cap_mb``,
            until n is big enough to cover the entire model size. The order
            of which buffer is ready there is more rigid and DDP requires all
            gradients to be computed in the backward. In FSDP, the buffer size
            does not change with model size (it changes based on number of
            <dtype, device, process_group> tuples) and gradient ready order matters
            little since FSDP has a final flush call that ensures everything is reduced
            and not all gradients need to be upfront known. Overlapping with compute is
            done differently too.
            Values <= 0 disable bucketing.
            Default: 25.
        compute_device (torch.device, Optional):
            device for computation. If not given and module params are on a CUDA
            device, the param's device will be used. If not given and module
            params are on CPU, then the current CUDA device (as indicated by
            ``torch.cuda.current_device()`` will be used.
        no_broadcast_optim_state: (bool, Optional)
            do not broadcast this modules optimizer state when ``gather_full_optim_state_dict`` is called.
            If you set this true, you are expected to overwrite the relevant state entries of the returned optimizer state dict
            with the proper state at each rank. This is useful for situations, like Mixture Of Experts,
            where all but a few parameters can fit on one node.
            Default: False
        state_dict_device (torch.device, Optional):
            device for parameters returned by :func:`state_dict`. If not given,
            this will default to ``compute_dtype``. Note that only the device
            type will be respected (e.g., "cuda:0" and "cuda:1" are the same).
        clear_autocast_cache (bool):
            When using mixed precision training with `torch.amp.autocast`, if the model weights
            are in FP32, autocast maintains a cache for downcasted weights. The cache can cause
            GPU OOM during the forward pass. Setting this flag to true will help clearing this
            cache as inner FSDP instances finish part of the forward pass to save GPU memory.
            Default: False
        force_input_to_fp32 (bool):
            Set to ``True`` to force input floating point tensors to be FP32 (if they are FP16)
            when the FSDP instance is in full precision mode. This helps avoid issues of running
            SyncBatchNorm with AMP and checkpoint_wrapper.
            Default: False
        verbose (bool):
            Set this to ``True`` to turn on verbose output for model's string representation.
            Default: False
        cpu_offload (bool, Optional):
            if ``True``, offload FP32 params to CPU. This is only relevant when
            *``mixed_precision``* is ``True``. Note: This arg will be deprecated in favor of
            *``move_params_to_cpu``* in an upcoming release.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        reshard_after_forward: bool = True,
        mixed_precision: bool = False,
        fp32_reduce_scatter: bool = False,
        flatten_parameters: bool = True,
        move_params_to_cpu: bool = False,
        compute_dtype: Optional[torch.dtype] = None,
        buffer_dtype: Optional[torch.dtype] = None,
        move_grads_to_cpu: Optional[bool] = None,
        bucket_cap_mb: int = 25,
        compute_device: Optional[torch.device] = None,
        no_broadcast_optim_state: Optional[bool] = False,
        state_dict_device: Optional[torch.device] = None,
        clear_autocast_cache: bool = False,
        force_input_to_fp32: bool = False,
        verbose: bool = False,
        cpu_offload: bool = False,
    ):
        init_start = time.time()
        super().__init__()
        self.process_group = process_group or get_process_group_cached()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        self.reshard_after_forward = reshard_after_forward
        self.mixed_precision = mixed_precision
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self.flatten_parameters = flatten_parameters
        self.move_params_to_cpu = move_params_to_cpu or cpu_offload
        self.compute_dtype = compute_dtype or (torch.float16 if mixed_precision else torch.float32)
        self.buffer_dtype = buffer_dtype or self.compute_dtype
        self.move_grads_to_cpu = self.move_params_to_cpu if move_grads_to_cpu is None else move_grads_to_cpu
        self.bucket_cap_mb = bucket_cap_mb
        self.compute_device = compute_device or _get_default_cuda_device(module)
        self.uncollected_opt_state: Dict[int, Dict] = {}
        self.no_broadcast_optim_state = no_broadcast_optim_state
        self.state_dict_device = state_dict_device or self.compute_device
        self.clear_autocast_cache = clear_autocast_cache
        self.force_input_to_fp32 = force_input_to_fp32
        self.verbose = verbose

        self.gradient_predivide_factor: float = self._get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

        self.numel_padded_per_param: List[int] = []
        self._tstart = time.time()

        if self.fp32_reduce_scatter and not self.mixed_precision:
            raise ValueError("fp32_reduce_scatter requires mixed_precision=True")
        if self.move_params_to_cpu and not self.mixed_precision:
            raise ValueError("cpu_offload requires mixed_precision=True")

        # skip validation if the process group was created above
        if process_group:
            validate_process_group(self.compute_device, self.process_group)

        # enable pytorch sync_bn just in case model contains sync_bn layers.
        enable_pytorch_sync_bn(module)

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        param_names = []
        params = []
        for param_name, param in module.named_parameters():
            if not hasattr(param, "_is_sharded"):
                param_names.append(param_name)
                params.append(param)

        self._has_params = len(params) > 0
        if not self._has_params:
            self.flatten_parameters = False

        if self.flatten_parameters:
            self._fsdp_wrapped_module: nn.Module = FlattenParamsWrapper(module, param_list=params)
            del module  # free original module in case it helps garbage collection
            self.param_paths = ["flat_param"]
            self.params = [self._fsdp_wrapped_module.flat_param]
        else:
            self._fsdp_wrapped_module = module
            self.param_paths = param_names
            self.params = params

        # Shard module parameters in place
        self._shard_parameters_()

        # Make sure all parameters are sharded.
        for n, p in self.named_parameters():
            assert hasattr(p, "_is_sharded"), f"found unsharded parameter: {n} ; {p.size()}"

        self._reset_lazy_init()

        # Flag to indicate if we require gradient reduction in the backward
        # pass. This will be False when inside the no_sync context manager.
        self._require_backward_grad_sync: bool = True

        # Enum to indicate if we're in the forward/backward pass, idle, etc.
        self.training_state = TrainingState.IDLE

        # Flag to indicate if the full params are gathered.
        self.has_full_params: bool = False

        # Register hook after state_dict() to remove the "_fsdp_wrapped_module."
        # prefix and before load_state_dict() to add it back.
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)

        # Flag to indicate whether state_dict() should automatically summon the
        # full params. This defaults to True, but may be set to False if the
        # user explicitly requests the local state dict via local_state_dict().
        self._return_full_state_dict = True
        init_end = time.time()

        logging.debug(
            f"FSDP.__init__(done): total_init_time: {(init_end - init_start): .4f} num_params: {(sum(p.numel() for p in self.params))}"
        )

        # Flag to guard multiple pre-backward hook being executed per iteration.
        # This is reset at the end of the backward pass.
        self._pre_backward_hook_has_run = False

    def _get_gradient_predivide_factor(self, world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)

    def set_gradient_divide_factors(self, pre: float, post: float, recursive: bool) -> None:
        """Allowing user to override the pre and post divide factors.

        Args:
            pre (float): divide factor before the reduction.
            post (float): divide factor after the reduction.
            recursive (bool): recursively set it for all child FSDP instances or not.
        """
        self.assert_state(TrainingState.IDLE)
        if recursive:
            for module in self.modules():
                if isinstance(module, FullyShardedDataParallel) and module != self:
                    module.set_gradient_divide_factors(pre, post, False)
        self.gradient_predivide_factor = pre
        self.gradient_postdivide_factor = post

    @property
    def module(self) -> nn.Module:
        return self._fsdp_wrapped_module  # note: may be a FlattenParamsWrapper instance

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        """
        Applies ``fn`` recursively to every submodule (as returned by
        ``.children()``) as well as self. Typical use includes initializing the
        parameters of a model.

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (nn.Module): function to be applied to each submodule

        Returns:
            Module: self
        """
        is_uninitialized = self._is_root is None
        self.assert_state(TrainingState.IDLE)
        with self.summon_full_params(recurse=False):
            return_value = super().apply(fn)
        # summon_full_params will call _lazy_init, which sets _is_root. However,
        # apply() may be called directly on children instances to do weight
        # init, so we should reset the _is_root flag in this case.
        if is_uninitialized and self._is_root:
            for module in self.modules():
                if isinstance(module, FullyShardedDataParallel):
                    module._reset_lazy_init()
        return return_value

    def _cast_buffers(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, memo: Optional[Set] = None
    ) -> None:
        """Move all buffers to the given *device* and *dtype*.

        If *device* or *dtype* are not given, then they will default to
        ``self.compute_device`` and ``self.buffer_dtype``, respectively. In the
        case of nested FSDP instances, we will respect the child instance's
        ``compute_device`` and ``buffer_dtype`` configuration.

        Args:
            device (torch.device, Optional):
                device to cast buffers to (defaults to compute_device)
            dtype (torch.dtype, Optional):
                dtype to cast buffers to (defaults to buffer_dtype)
            memo (Set, Optional):
                set of modules that have already been processed
        """
        if memo is None:
            memo = set()
        for module in self.modules():
            if module is not self and isinstance(module, FullyShardedDataParallel):
                # Allow any child FSDP instances to handle their own buffers.
                module._cast_buffers(device=device, dtype=dtype, memo=memo)
            elif module not in memo:
                memo.add(module)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is None:
                        continue
                    buf = buf.to(device=device or self.compute_device)
                    if torch.is_floating_point(buf):
                        buf = buf.to(dtype=dtype or self.buffer_dtype)
                    setattr(module, name, buf)

    @property
    def params_with_grad(self) -> List[Parameter]:
        """[p for p in self.parameters() if p.grad is not None] """
        return [p for p in self.parameters() if p.grad is not None]

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        # filter_params_fn: Callable[[Any], Any] = None,
    ) -> torch.Tensor:
        """
        Clip all gradients at this point in time. The norm is computed over all
        gradients together, as if they were concatenated into a single vector.
        Gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note:: This is analogous to `torch.nn.utils.clip_grad_norm_` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it in the OSS context would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        # We don't call torch.cuda.synchronize() here, since clipping can be
        # inside the train loop and we probably don't want to force a GPU-CPU sync.
        # _lazy_init should be sufficient, since it will force the other streams
        # to sync with the default stream (via _wait_for_previous_optim_step).
        self._lazy_init()
        assert self._is_root, "clip_grad_norm should only be called on the root (parent) instance"
        self.assert_state(TrainingState.IDLE)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        params_with_grad = self.params_with_grad
        if not self.children_share_process_group:
            raise NotImplementedError(
                "clip_grad_norm requires that all params share one process group. clip_grad_by_value_ should work"
            )
        # Computes the max norm for this shard's gradients and sync's across workers
        local_norm = calc_grad_norm(params_with_grad, norm_type).cuda()
        if norm_type == inf:
            total_norm = local_norm
            dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
        else:
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            total_norm = total_norm ** (1.0 / norm_type)

        if self.move_grads_to_cpu:
            total_norm = total_norm.cpu()
        # Now multiply each grad by (max_norm/total_norm), same as torch 1.7 https://tinyurl.com/3wtxhhqq)
        clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-6)
        if clip_coef < 1:

            # multiply by clip_coef
            for p in params_with_grad:
                p.grad.detach().mul_(clip_coef.to(p.grad.device))  # type: ignore

        return total_norm

    @torch.no_grad()
    def _shard_parameters_(self) -> None:
        """
        At initialization we wrap a module with full parameters and shard the
        parameters in-place. Sharding is implemented by viewing each parameter
        as a 1D Tensor and retaining only a single slice, where the slice size
        is determined by the number of data parallel workers.

        Wrapping modules with many small parameters (or with a very large data
        parallel world size) will result in many small parameter shards and slow
        performance. In this case it's better to set *``flatten_parameters``* to
        ``True``, so that all of the small parameters in the module are combined
        into a single contiguous Tensor and sharded once.

        After this initial sharding is complete, the user can initialize a
        ``torch.optim.Optimizer`` in the usual way, i.e.::

        .. code-block:: python

            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

        The optimizer will see only a single slice of parameters and will thus
        allocate less memory for optimizer state, avoiding redundancy across
        data parallel workers.
        """
        self.numel_padded_per_param = []
        for p in self.params:
            assert not hasattr(p, "_is_sharded")
            assert p.is_floating_point()
            if self.mixed_precision:
                assert p.dtype == torch.float32

            # If world_size is 1, then we all-reduce grads instead of sharding.
            p._is_sharded = self.world_size > 1
            p._orig_size = p.data.size()

            if not p._is_sharded:
                self.numel_padded_per_param.append(0)
                continue
            p._is_sharded = True

            # Replace p.data with the relevant shard.
            orig_data = p.data
            p.data, num_padded = self._get_shard(p.data)
            self.numel_padded_per_param.append(num_padded)
            free_storage_(orig_data)
        assert len(self.numel_padded_per_param) == len(self.params)

    def _get_shard(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Return the local shard of a full tensor."""
        # Shard using torch.chunk to match all-gather/reduce-scatter.
        chunks = list(torch.flatten(tensor).chunk(self.world_size))
        while len(chunks) < self.world_size:
            chunks.append(chunks[0].new_empty(0))

        # Determine number of padding elements.
        num_to_pad = chunks[0].numel() - chunks[self.rank].numel()
        assert num_to_pad >= 0, num_to_pad

        shard = chunks[self.rank].clone()
        if num_to_pad > 0:
            shard = F.pad(shard, [0, num_to_pad])
        return shard, num_to_pad

    def extra_repr(self) -> str:
        repr = (
            f"world_size={self.world_size}, "
            f"flatten_parameters={self.flatten_parameters}, "
            f"mixed_precision={self.mixed_precision}, "
        )
        if self.verbose:
            repr = (
                f"rank={self.rank}, " + repr + f"reshard_after_forward={self.reshard_after_forward}, "
                f"compute_dtype={self.compute_dtype}, "
                f"buffer_dtype={self.buffer_dtype}, "
                f"fp32_reduce_scatter={self.fp32_reduce_scatter}, "
                f"compute_device={self.compute_device}"
                f"cpu_offload={self.move_params_to_cpu}, "
                f"move_grads_to_cpu={self.move_grads_to_cpu}, "
                f"bucket_cap_mb={self.bucket_cap_mb}, "
                f"clear_autocast_cache={self.clear_autocast_cache}"
                f"force_input_to_fp32={self.force_input_to_fp32}"
            )
        return repr

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def __getstate__(self) -> Dict[str, str]:
        """Serialize the state of the current FullyShardedDataParallel instance.

        Some properties are not serializable (e.g., process groups, streams), so
        we remove them and try to reconstruct them in :func:`__setstate__`.
        """
        state = copy.copy(self.__dict__)
        state["is_sharded"] = [p._is_sharded for p in self.params]
        state["orig_sizes"] = [p._orig_size for p in self.params]
        if state["process_group"] is not None:
            state["process_group"] = "MISSING"  # process_group isn't pickleable
        self._reset_lazy_init()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Intercept state setting and perform needed changes on params."""
        super().__setstate__(state)

        def fixup(p: Parameter, is_sharded: bool, size: torch.Size) -> Parameter:
            assert isinstance(p, Parameter)
            p.data = p.data.clone()  # move tensors out of shared memory
            p._is_sharded = is_sharded
            p._orig_size = size
            return p

        self.params = [
            fixup(p, is_sharded, size) for p, is_sharded, size in zip(self.params, self.is_sharded, self.orig_sizes)
        ]
        del self.is_sharded
        del self.orig_sizes
        self._reset_lazy_init()

    @typing.overload
    def state_dict(
        self, destination: Mapping[str, torch.Tensor], prefix: str = ..., keep_vars: bool = ...
    ) -> Mapping[str, torch.Tensor]:
        ...

    @typing.overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> "OrderedDict[str, torch.Tensor]":
        ...

    # Since we have overloads above, we can use Any here.
    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the whole (unsharded) state of the module. Parameters are not
        sharded, so the resulting state_dict can be loaded directly by the
        wrapped Module without any sharding-specific logic. Returned tensors
        will be full precision (e.g., FP32).

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._lazy_init()

        def maybe_cast_buffers(dtype: Optional[torch.dtype] = None) -> None:
            if self.mixed_precision:
                self._cast_buffers(dtype=dtype)

        if self._return_full_state_dict:
            if self.training_state != TrainingState.SUMMON_FULL_PARAMS:
                with self.summon_full_params(recurse=False, volatile=True):
                    maybe_cast_buffers(torch.float32)
                    state_dict = super().state_dict(*args, **kwargs)
            else:
                maybe_cast_buffers(torch.float32)
                state_dict = super().state_dict(*args, **kwargs)
        else:
            maybe_cast_buffers(torch.float32)
            if self.flatten_parameters:
                assert isinstance(self.module, FlattenParamsWrapper)
                state_dict = self.module.flat_state_dict(*args, **kwargs)
            else:
                state_dict = super().state_dict(*args, **kwargs)

        if self.move_params_to_cpu:
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].cpu()

        # In case we are in mixed precision, restore buffers back to buffer_dtype.
        maybe_cast_buffers()
        return state_dict

    @typing.overload
    def local_state_dict(
        self, destination: Mapping[str, torch.Tensor], prefix: str = ..., keep_vars: bool = ...
    ) -> Mapping[str, torch.Tensor]:
        ...

    @typing.overload
    def local_state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> "OrderedDict[str, torch.Tensor]":
        ...

    # Since we have overloads above, we can use Any here.
    def local_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the local (sharded) state of the module. Parameters are sharded,
        so the resulting state_dict can only be loaded after the Module has been
        wrapped with FullyShardedDataParallel.
        """
        with contextlib.ExitStack() as stack:
            # Tell any nested FSDP instances not to auto summon full params.
            for module in self.modules():  # includes self
                if isinstance(module, FullyShardedDataParallel):
                    stack.enter_context(module._no_return_full_state_dict())
            # We need to specially call FSDP's state_dict function in case
            # self.state_dict is a function from a child class of FSDP.
            return FullyShardedDataParallel.state_dict(self, *args, **kwargs)

    @contextlib.contextmanager
    def _no_return_full_state_dict(self) -> Generator:
        backup = self._return_full_state_dict
        self._return_full_state_dict = False
        try:
            yield
        finally:
            self._return_full_state_dict = backup

    def _load_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """
        Load a whole (unsharded) state_dict.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        if self._return_full_state_dict:
            with self.summon_full_params():
                return self.module.load_state_dict(state_dict, strict)
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            return self.module.load_state_dict(state_dict, strict)

    def load_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        return self._load_state_dict(state_dict, strict)

    def load_local_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """Load a local (sharded) state_dict."""
        with contextlib.ExitStack() as stack:
            # Tell any nested FSDP instances not to auto summon full params.
            for module in self.modules():  # includes self
                if isinstance(module, FullyShardedDataParallel):
                    stack.enter_context(module._no_return_full_state_dict())
            output = self._load_state_dict(state_dict, strict)
        return output

    @contextlib.contextmanager
    def no_sync(self) -> Generator:
        """
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context.

        .. note:: This may result in higher memory usage because we will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.
        """
        self._lazy_init()
        assert self._is_root, "no_sync on inner FSDP is not supported"
        self.assert_state(TrainingState.IDLE)
        # This instance may wrap other FullyShardedDataParallel instances and we
        # need to set all of them to accumulate gradients.
        old_flags = []
        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel):
                old_flags.append((m, m._require_backward_grad_sync))
                m._require_backward_grad_sync = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                m._require_backward_grad_sync = old_flag

    @contextlib.contextmanager
    def summon_full_params(self, recurse: bool = True, volatile: bool = False) -> Generator:
        """
        A context manager to expose full params for the current FSDP instance.
        Can be useful *after* forward/backward for a model to get the params for
        additional processing or checking. Parameters will be gathered in full
        precision (e.g., FP32).

        .. note:: This can be used on inner FSDPs.

        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.

        .. note:: The full parameters will be freed after the context manager
            exits; it is up to the caller to clone them if needed.

        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``volatile=True``, in which case there
            are no guarantees about persistence).

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True)
            volatile (bool, Optional): if ``True``, modifications to params are
                not guaranteed to persist after the context manager exists;
                enabling this can be slightly more efficient (default: False)
        """
        if recurse:
            with contextlib.ExitStack() as stack:
                # Summon all params for any nested FSDP instances.
                for module in self.modules():
                    if isinstance(module, FullyShardedDataParallel):
                        stack.enter_context(module.summon_full_params(recurse=False, volatile=volatile))
                # Yield to the caller, with full params in all nested instances.
                yield
            # Exiting from the ExitStack will re-shard params.
            return
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            self.assert_state(TrainingState.IDLE)
            # Set the state so that we assert when trying to go into
            # forward/backward.
            self.training_state = TrainingState.SUMMON_FULL_PARAMS
            full_tensors = self._rebuild_full_params(force_full_precision=True)
            assert full_tensors is not None
            with contextlib.ExitStack() as stack:
                if self.flatten_parameters and self.module.is_flattened:
                    # Update flattened views to point to fully-sized tensors. We
                    # use self.params[0] instead of full_tensors since the
                    # latter may contain padding.
                    assert len(self.params) == 1
                    assert isinstance(self.module, FlattenParamsWrapper)
                    stack.enter_context(self.module.unflatten_params(flat_params=[self.params[0]]))
                try:
                    yield
                finally:
                    stack.close()
                    assert len(full_tensors) == len(self.params)
                    for p, (full_tensor, safe_to_free) in zip(self.params, full_tensors):
                        if not volatile:
                            # Copy any changes made to the full params back into
                            # the corresponding local shards.
                            local_shard, _ = self._get_shard(full_tensor)
                            p._fp32_shard.copy_(local_shard.view_as(p._fp32_shard))
                        if safe_to_free:
                            free_storage_(full_tensor)
                    self.has_full_params = False
                    self._use_fp32_param_shard()
                    self.training_state = TrainingState.IDLE

    def _reset_lazy_init(self) -> None:
        """Reset instance so :func:`_lazy_init` will run on the next forward."""
        self._is_root: Optional[bool] = None
        self._queue_wait_for_post_backward_closure: Optional[Callable] = None
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._reducer: Optional[ReduceScatterBucketer] = None
        for p in self.params:
            if hasattr(p, "_fp32_shard"):
                del p._fp32_shard  # reset _init_param_attributes

    def _lazy_init(self) -> None:
        """Initialization steps that should happen lazily, typically right
        before the first forward pass.
        """
        # Initialize param attributes lazily, in case the param's dtype or
        # device changes after __init__.
        for p in self.params:
            self._init_param_attributes(p)

        # Initialize _is_root and setup streams. These steps would ideally
        # happen in __init__, but _is_root can only be determined after the
        # entire model hierarchy is setup, thus we run it lazily.
        if self._is_root is None:
            self._set_is_root()
            self._setup_streams()

        if self._is_root:
            # Buffers stay on GPU, and don't get sharded. Since _cast_buffers
            # applies recursively, we only call this from the root instance.
            self._cast_buffers()

            # Don't free the full params for the outer-most (root) instance,
            # since those params will be needed immediately after for the
            # backward pass.
            self.reshard_after_forward = False

            # Due to the use of streams, we need to make sure the previous
            # ``optim.step()`` is done before we all-gather parameters.
            self._wait_for_previous_optim_step()

    @torch.no_grad()
    def _init_param_attributes(self, p: Parameter) -> None:
        """
        We manage several attributes on each Parameter instance. The first two
        are set by :func:`_shard_parameters_`:

            ``_is_sharded``: ``True`` if the Parameter is sharded or ``False``
                if the Parameter is intentionally not sharded (in which case we
                will all-reduce grads for this param).
            ``_orig_size``: the size of the original Parameter (before sharding)

        The remaining attributes are set here:
            ``_fp32_shard``: a single shard of the parameters in full precision
                (typically FP32, but this is dependent on the dtype of the model
                as it's passed in by the user). This can be on CPU or GPU
                depending on the value of *``cpu_offload``*.
            ``_fp16_shard``: if *``mixed_precision``* is ``True``, this will be
                a single shard of the parameters in FP16, used for all-gather.
            ``_full_param_padded``: the full weight (padded to be evenly
                divisible by ``world_size``), used for computation in the
                forward and backward pass. This will be resized in place and
                only materialized (via all-gather) as needed.
        """
        assert hasattr(p, "_is_sharded") and hasattr(p, "_orig_size")
        if hasattr(p, "_fp32_shard"):
            return

        # A single shard of the parameters in full precision.
        p._fp32_shard = p.data

        if self.mixed_precision:
            assert p._fp32_shard.dtype == torch.float32

            if self.move_params_to_cpu:
                assert p._fp32_shard.device == torch.device("cpu")
                # If we plan to keep the FP32 parameters on CPU, then pinning
                # memory allows us to later use non-blocking transfers when moving
                # the FP32 param shard to compute_device.
                p._fp32_shard = p._fp32_shard.pin_memory()
                p.data = p._fp32_shard

            # In mixed precision mode, we maintain a reduced precision
            # (typically FP16) parameter shard on compute_device for performing
            # the computation in the forward/backward pass. We resize the
            # storage to size 0 at init (here) and re-materialize (by copying
            # from _fp32_shard) as needed.
            p._fp16_shard = torch.zeros_like(p._fp32_shard, device=self.compute_device, dtype=self.compute_dtype)
            free_storage_(p._fp16_shard)
        else:
            p._fp16_shard = None  # use _fp32_shard

        # We also maintain a full-sized parameter of type self.compute_dtype
        # (FP16 for mixed_precision or FP32 otherwise). We resize the
        # storage to size 0 at init (here) and only materialize as needed. The
        # storage may contain padding elements so that it is evenly divisible by
        # world_size, although these padding elements will be removed before the
        # relevant computation.
        if p._is_sharded:
            p._full_param_padded = torch.zeros(
                p.data.numel() * self.world_size, device=self.compute_device, dtype=self.compute_dtype
            )
            free_storage_(p._full_param_padded)

        if self.move_grads_to_cpu:
            # We can optionally move the grad shard to CPU during the backward
            # pass. In this case, it's important to pre-allocate the CPU grad
            # shard in pinned memory so that we can do a non-blocking transfer.
            p._cpu_grad = torch.zeros_like(p.data, device="cpu").pin_memory()

    def _set_is_root(self) -> None:
        """If ``True``, implies that no other :class:`FullyShardedDataParallel`
        instance wraps this one. Called once by :func:`_lazy_init`.
        Also sets self.children_share_process_group = True if all child
        instances share the same process group. If some child instances use a
        different process group, self.clip_grad_norm_ will raise an error.
        """
        if self._is_root is not None:
            return
        # No FullyShardedDataParallel instance wraps this, else _is_root would be set to False.
        self._is_root = True
        assert self._queue_wait_for_post_backward_closure is None
        self._queue_wait_for_post_backward_closure = self._queue_wait_for_post_backward
        # As the root, we now set all children instances to False and
        # give them a closure to try to queue a wait_for_post_backward.
        self.children_share_process_group = True
        for n, m in self.named_modules():
            # `n != ""` excludes self.
            if n != "" and isinstance(m, FullyShardedDataParallel):
                # We relax the assert for non-root instance, when the nested inialized module is wrapped
                # again in FSDP later, for example after training to run inference.
                assert m._is_root is None or not m._is_root
                if m._is_root is None:
                    m._is_root = False
                    # When root instance doesn't have params, allow children instances
                    # to queue the post_backward hook.
                    #
                    # TODO (Min): we should think if we can have a empty param at the root
                    #             so that root always have a callback on the backward graph.
                    if not self._has_params:
                        assert m._queue_wait_for_post_backward_closure is None
                        m._queue_wait_for_post_backward_closure = self._queue_wait_for_post_backward
                if m.process_group != self.process_group:
                    self.children_share_process_group = False

                # if child instance in its own (smaller) world, that was probably an attempt to avoid OOM.
                # Therefore gathering this child's optim state will probably cause OOM, so we won't do it.
                m.no_broadcast_optim_state = m.no_broadcast_optim_state or (
                    (m.world_size == 1) and (m.world_size < self.world_size) and (m.process_group != self.process_group)
                )

    def _setup_streams(self) -> None:
        """Create streams to overlap data transfer and computation."""
        if len(self._streams) > 0 or not self._is_root:
            return

        if torch.cuda.is_available():
            # Stream to move main FP32 params (may be on CPU) to FP16 for forward.
            self._streams["fp32_to_fp16"] = torch.cuda.Stream()
            # Stream for all-gathering parameters.
            self._streams["all_gather"] = torch.cuda.Stream()
            # Stream for overlapping grad reduction with the backward pass.
            self._streams["post_backward"] = torch.cuda.Stream()

        # Helper for bucketing reduce-scatter ops. This is also shared with
        # children instances to improve bucket utilization.
        self._reducer = ReduceScatterBucketer(self.bucket_cap_mb)
        # We share streams with all children instances, which allows them to
        # overlap transfers across the forward pass without synchronizing with
        # the default stream.
        for n, m in self.named_modules():
            if n != "" and isinstance(m, FullyShardedDataParallel):
                m._streams = self._streams
                m._reducer = self._reducer

    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`FullyShardedDataParallel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if not torch.cuda.is_available():
            return
        if self.mixed_precision:
            self._streams["fp32_to_fp16"].wait_stream(torch.cuda.current_stream())
        else:
            self._streams["all_gather"].wait_stream(torch.cuda.current_stream())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._lazy_init()

        # Start of a forward pass.
        self.training_state = TrainingState.FORWARD

        # For root and mixed precision, we convert the input to FP16 (no_grad is needed for
        # the conversion).
        if self._is_root and self.mixed_precision:
            args, kwargs = cast_floats_to_right_precision(True, True, *args, **kwargs)

        # If enabled, convert the input to FP32 if we are in full precision.
        # no_grad is not used because the input might be for a non-root instance,
        # which mean autograd needs to go through the conversion.
        if self.force_input_to_fp32 and not self.mixed_precision:
            args, kwargs = cast_floats_to_right_precision(False, False, *args, **kwargs)

        # All-gather full parameters. This will also transfer FP32 parameters to
        # ``self.compute_dtype`` (e.g., FP16 if *mixed_precision* is ``True``).
        self._rebuild_full_params()

        # Register backward hooks to reshard params and reduce-scatter grads.
        # These need to be re-registered every forward pass.
        self._register_post_backward_hooks()

        outputs = self.module(*args, **kwargs)

        if self.reshard_after_forward:
            self._free_full_params()
            if self.mixed_precision:
                self._free_fp16_param_shard()

        # Switch to main FP32 param shard. We maintain this invariant throughout
        # the code, i.e., ``p.data == p._fp32_shard`` after each function. This
        # also ensures that after the first forward, the optimizer state will be
        # initialized with the correct dtype and (sharded) size, since optimizer
        # state is typically initialized lazily in ``optim.step()``.
        self._use_fp32_param_shard()

        # Register pre-backward hooks to all-gather the params for the backward
        # pass (if output's grad was needed). This won't register anything if
        # we are in eval mode.
        #
        # Some model does forward pass multiple times, we need to register the
        # pre-backward hook on every output since the last output's hook has to
        # fire first to setup for backward. However, we use ``self._pre_backward_hook_has_run``
        # to prevent repeated overhead from multiple hook callbacks.
        outputs = self._register_pre_backward_hooks(outputs)

        # Done with a forward pass.
        self.training_state = TrainingState.IDLE

        # Only need to clear cache during forward. During backward, the cache is not used.
        # TODO (Min): Future PyTorch versions may provide a way to completely disable this
        #     cache. Update this when that's available.
        if self.clear_autocast_cache:
            torch.clear_autocast_cache()

        return outputs

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:
        """Register pre-backward hook to run before the wrapped module's
        backward. Hooks should be attached to all outputs from the forward.

        Returns:
            outputs: new outputs with hooks registered if they requires gradient.
        """
        if not torch.is_grad_enabled():
            return outputs  # don't register hooks if grad isn't enabled

        def _pre_backward_hook(*unused: Any) -> None:
            if self._pre_backward_hook_has_run:
                return  # only run once (from multiple outputs or multiple forward passes)
            self._pre_backward_hook_has_run = True

            # Start of a backward pass.
            self.assert_state([TrainingState.IDLE, TrainingState.BACKWARD_PRE])
            self.training_state = TrainingState.BACKWARD_PRE

            # All-gather full parameters.
            if self.reshard_after_forward:
                self._rebuild_full_params()
            else:
                self._use_full_params()

            # Make sure p.grad has the correct size/device (or set it to None).
            self._prep_grads_for_backward()

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            if t.requires_grad:
                t.register_hook(_pre_backward_hook)
            return t

        # Attach hooks to Tensor outputs.
        outputs = apply_to_tensors(_register_hook, outputs)

        return outputs

    def _register_post_backward_hooks(self) -> None:
        """
        Register backward hooks to reshard params and reduce-scatter grads.

        This is called during forward pass. The goal is to attach a hook
        on each of the parameter's gradient generating function (``grad_acc``
        below) so that the hook is called *after* all gradients for that
        param are computed.

        Goals:

        1. We want the hook to fire once and only once *after* all gradients
        are accumulated for a param.
        2. If it fires more than once, we end up incorrectly shard the grad
        multiple times. (could lead to dimension too small)
        3. If it fires once but too early or doesn't fire, we leave gradients
        unsharded. (could lead to dimension too large)

        Due to multiple-pass forward, this function can be called on
        the same parameter multiple times in a single forward pass. If we register
        the hook multiple time, we end up getting called multiple times. We
        could try to get a new hook every time and delete the previous one
        registered. However, due to *unknown reason* (I have debugged it for
        a long time!), in mixed precision mode, we get two different ``grad_acc``
        objects below during different calls of this function (in the same
        forward pass). If we keep the last one, the hook end up firing too
        early. In full precision mode, we luckily get the *same* ``grad_acc``
        object, so deleting and re-registering still ensured the hook fire
        once after all gradients are generated.

        Empirically, keep the first hook register per forward pass seems to
        work the best. We do need to remove the hook at the end of the
        backward pass. Otherwise, the next forward pass will not register
        a new hook, which is needed for a new forward pass.
        """
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        if self._is_root:
            # This actually means that only root instance has this field
            # defined. Accidentally accessing this field will assert on all
            # other instances, giving us a nice bug checker.
            self._post_backward_callback_queued = False
        for p in self.params:
            if p.requires_grad:
                if hasattr(p, "_shard_bwd_hook"):
                    continue
                # Register a hook on the first call, empirically, autograd
                # fires it at the end for this param, which makes sense.
                p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                assert p_tmp.grad_fn is not None
                grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
                handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p))
                p._shard_bwd_hook = (grad_acc, handle)

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
        """
        At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will replace
        ``param.grad`` with a single shard of the summed gradient across all
        GPUs. This shard will align with the current GPU rank. For example::

            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]

            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8

        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by :func:`_shard_parameters_`, which ensures that
        the local optimizer only sees the relevant parameter shard.
        """
        # First hook callback will see PRE state. If we have multiple params,
        # then subsequent hook callbacks will see POST state. When checkpoint
        # fwd counter is used, IDLE is also possible since the pre-backward hook
        # is not triggered (see ``auto_wrap_bn`` below, we have to use
        # FSDP(checkpoint(conv, FSDP(bn), ...)), with reshard_after_forward=False).
        if hasattr(self, "_checkpoint_fwd_counter"):
            self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST, TrainingState.IDLE])
        else:
            self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
        self.training_state = TrainingState.BACKWARD_POST
        if param.grad is None:
            return

        if param.grad.requires_grad:
            raise RuntimeError("FullyShardedDataParallel only works with gradients that don't require gradients")

        # If this is a checkpointed module, we check if the following
        # counter reaches 0. If not, it is not the final backward call
        # for this module yet. Therefore, we early return in that case.
        if hasattr(self._fsdp_wrapped_module, "_checkpoint_fwd_counter"):
            if self._fsdp_wrapped_module._checkpoint_fwd_counter != 0:
                return

        if self._require_backward_grad_sync or self.reshard_after_forward:
            # Free full params. As a special case, we don't free the full params
            # when in a ``no_sync`` context (as inversely indicated by
            # ``self._require_backward_grad_sync``), since the params will not
            # get updated before the next forward.
            self._free_full_params([param])

        if self.mixed_precision:
            # This is a no-op if reshard_after_forward is True, since we already
            # free the param shard when rebuilding the full params in the
            # pre_backward_hook.
            self._free_fp16_param_shard([param])

        # Switch to FP32 shard after backward.
        self._use_fp32_param_shard([param])

        # (try to) Enqueue a callback at the end of the backward pass to ensure that all
        # post-backward work has finished. We only need one callback and all instances
        # of FSDP (root and children) make this attempt here to queue to ensure it is queued
        # no matter which instance(s) has(have) params.
        assert self._queue_wait_for_post_backward_closure is not None or not self._is_root
        if self._queue_wait_for_post_backward_closure is not None:
            self._queue_wait_for_post_backward_closure()

        if not self._require_backward_grad_sync:
            return

        # Wait for all work in the current stream to finish, then start the
        # reductions in post_backward stream.
        self._streams["post_backward"].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._streams["post_backward"]):
            orig_grad_data = param.grad.data

            if self.mixed_precision and self.fp32_reduce_scatter:
                # Cast grad to FP32.
                param.grad.data = param.grad.data.to(param.dtype)

            if self.gradient_predivide_factor > 1:
                # Average grad by world_size for consistency with PyTorch DDP.
                param.grad.data.div_(self.gradient_predivide_factor)

            callback_fn = functools.partial(self._post_reduction_hook, param)
            if param._is_sharded:
                assert param._is_sharded
                assert self._reducer is not None
                grad_chunks = chunk_and_pad(param.grad.data, self.world_size)
                self._reducer.reduce_scatter_async(grad_chunks, group=self.process_group, callback_fn=callback_fn)
            else:
                # Currently the only way for _is_sharded to be False is if
                # world_size == 1. This could be relaxed in the future, in which
                # case grads should be all-reduced here.
                assert self.world_size == 1
                callback_fn(param.grad.data)

            # After _post_backward_hook returns, orig_grad_data will eventually
            # go out of scope, at which point it could otherwise be freed for
            # further reuse by the main stream while the div/reduce_scatter/copy
            # are underway in the post_backward stream. See:
            # github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py
            orig_grad_data.record_stream(self._streams["post_backward"])

    def _post_reduction_hook(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        """Hook to call on each param after the reduce-scatter."""
        assert torch.cuda.current_stream() == self._streams["post_backward"]
        assert param.grad is not None
        self.assert_state(TrainingState.BACKWARD_POST)
        param.grad.data = reduced_grad
        if self.gradient_postdivide_factor > 1:
            # Average grad by world_size for consistency with PyTorch DDP.
            param.grad.data.div_(self.gradient_postdivide_factor)
        # Cast grad to param's dtype (typically FP32). Note: we do this
        # before the move_grads_to_cpu step so that this entire hook remains
        # non-blocking. The downside is a bit more D2H transfer in that case.
        if self.mixed_precision:
            orig_param_grad_data = param.grad.data
            param.grad.data = param.grad.data.to(dtype=param.data.dtype)
            # Don't let this memory get reused until after the transfer.
            orig_param_grad_data.record_stream(torch.cuda.current_stream())
        # Optionally move gradients to CPU, typically used if one is running
        # the optimizer on the CPU.
        if self.move_grads_to_cpu:
            param._cpu_grad.copy_(param.grad.data, non_blocking=False)
            # Don't let this memory get reused until after the transfer.
            param.grad.data.record_stream(torch.cuda.current_stream())
            param.grad.data = param._cpu_grad

    def _queue_wait_for_post_backward(self) -> None:
        """Try to queue a `wait_for_post_backward` callback.

        Only called on root and only queue one callback. But can be called by
        children FSDPs via a closure in case the root instance doesn't own any
        params.
        """
        assert self._is_root
        self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
        if not self._post_backward_callback_queued:
            self._post_backward_callback_queued = True
            Variable._execution_engine.queue_callback(self._wait_for_post_backward)

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward to finish. Only called on root instance."""
        assert self._is_root
        if self._has_params:
            self.assert_state(TrainingState.BACKWARD_POST)
        else:
            self.assert_state(TrainingState.BACKWARD_PRE)

        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self._streams["post_backward"]):
                assert self._reducer is not None
                self._reducer.flush()
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.move_grads_to_cpu:
                # Wait for the non-blocking GPU -> CPU grad transfers to finish.
                torch.cuda.current_stream().synchronize()

        # A backward pass is done, clean up below.

        # Free reducer buffers.
        if self._reducer is not None:
            self._reducer.teardown()

        def _remove_shard_bwd_hook(fsdp_module: FullyShardedDataParallel) -> None:
            """Helper used below on all fsdp modules."""
            for p in fsdp_module.params:
                if p.requires_grad:
                    if hasattr(p, "_shard_bwd_hook"):
                        assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
                        p._shard_bwd_hook[1].remove()
                        delattr(p, "_shard_bwd_hook")

        # Update root and nested FSDP's hooks and flags.
        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel):
                _remove_shard_bwd_hook(m)
                m._pre_backward_hook_has_run = False
                if any(p.requires_grad for p in m.parameters()):
                    if m._has_params:
                        m.assert_state(TrainingState.BACKWARD_POST)
                    else:
                        m.assert_state(TrainingState.BACKWARD_PRE)
                else:
                    # Unlikely case. When `m` and its children has no params or has params but
                    # none with `requires_grad==True`, then m's pre-backward and post-backward
                    # hooks aren't called by autograd. Therefore, it is in IDLE state.
                    m.assert_state(TrainingState.IDLE)
                m.training_state = TrainingState.IDLE

    @torch.no_grad()
    def _rebuild_full_params(self, force_full_precision: bool = False) -> Optional[List[Tuple[torch.Tensor, bool]]]:
        """
        Gather all shards of params.

        Args:
            force_full_precision (bool, Optional): by default params will be gathered
                in ``compute_dtype`` (e.g., FP16), unless *force_full_precision* is
                ``True``, in which case they will be gathered in full precision
                (e.g., FP32), possibly in fresh storage. The parameter that's being
                rebuilt will end up in full precision as well.

        Returns:
            A list of tuples, where the first element is the full-sized param
            and the second element is a bool indicating if it's safe for the
            caller to free the full-sized param. This will be ``None`` if
            ``force_full_precision=False`` and the full params are already gathered.
        """
        output_tensors: List[Tuple[torch.Tensor, bool]] = []

        def update_p_data(custom_output_tensor: Optional[torch.Tensor] = None) -> None:
            """
            Helper function to update p.data pointer.

            Args:
                custom_output_tensor (torch.Tensor, Optional): if not None, this
                tensor contains the data we just gathered.
            """
            if custom_output_tensor is not None:
                assert p._is_sharded
                p.data = custom_output_tensor
                output_tensors.append((p.data, True))
            elif not p._is_sharded:
                if self.mixed_precision and not force_full_precision:
                    assert p._fp16_shard is not None
                    p.data = p._fp16_shard
                    output_tensors.append((p.data, True))
                else:
                    # Here p.data == p._fp32_shard, so it's not safe to free.
                    output_tensors.append((p.data, False))
            else:
                p.data = p._full_param_padded
                output_tensors.append((p.data, True))
            # Trim any padding and reshape to match original size.
            p.data = p.data[: p._orig_size.numel()].view(p._orig_size)

        # Early exit if we already have full params and don't need full precision.
        if self.has_full_params and not force_full_precision:
            for p in self.params:
                update_p_data()
            return output_tensors

        self.has_full_params = True

        with torch.cuda.stream(self._streams["all_gather"]):
            if self.mixed_precision and not force_full_precision:
                self._cast_fp32_param_shards_to_fp16()

            for p in self.params:
                if not p._is_sharded:  # e.g., when world_size == 1
                    update_p_data()
                else:
                    # If self.move_params_to_cpu and force_full_precision, we need to cast
                    # the FP32 CPU param to CUDA for the all-gather.
                    p_data = p.data.to(p._full_param_padded.device)

                    p_size = p._full_param_padded.size()
                    assert p_size.numel() % self.world_size == 0
                    if self.mixed_precision and force_full_precision:
                        # Allocate fresh tensor in full precision since we are in
                        # mixed precision and full precision rebuild is asked.
                        output_tensor = p_data.new_zeros(p_size)
                    else:
                        if p._full_param_padded.storage().size() != p_size.numel():
                            # Allocate based on full size from all shards.
                            alloc_storage_(p._full_param_padded, size=p_size)
                        output_tensor = p._full_param_padded

                    # Fill output_tensor with (p.data for each shard in self.world_size)
                    chunks = list(output_tensor.chunk(self.world_size))
                    dist.all_gather(chunks, p_data, group=self.process_group)

                    # Set p.data = output_tensor (with padding trimmed)
                    update_p_data(output_tensor)

                    if self.mixed_precision and not force_full_precision:
                        self._free_fp16_param_shard([p])
        torch.cuda.current_stream().wait_stream(self._streams["all_gather"])
        return output_tensors

    @torch.no_grad()
    def _use_full_params(self) -> None:
        """
        Switch p.data pointers to use the full params.

        Note: this assumes full params are already gathered.
        """
        assert self.has_full_params
        for p in self.params:
            if not p._is_sharded:
                if self.mixed_precision:
                    assert p._fp16_shard is not None
                    assert p._fp16_shard.storage().size() != 0
                    p.data = p._fp16_shard
            else:
                assert p._full_param_padded.storage().size() != 0
                p.data = p._full_param_padded[: p._orig_size.numel()].view(p._orig_size)

    @torch.no_grad()
    def _prep_grads_for_backward(self) -> None:
        """Make sure p.grad has the correct size/device, otherwise set it to None."""
        for p in self.params:
            if p.grad is not None and (p.grad.size() != p._orig_size or p.grad.device != p.data.device):
                p.grad = None

    @torch.no_grad()
    def _free_full_params(self, params: Optional[List[Parameter]] = None) -> None:
        """Free up storage for full parameters."""
        if params is None:
            params = self.params
        self.has_full_params = False
        current_stream = torch.cuda.current_stream()
        for p in params:
            if not p._is_sharded:  # e.g., world_size == 1
                if self.mixed_precision:
                    self._free_fp16_param_shard([p])
                continue
            # Don't let PyTorch reuse this memory until all work in the current
            # stream is complete.
            p._full_param_padded.record_stream(current_stream)
            # There may be external references to the Tensor Storage that we
            # can't modify, such as references that are created by
            # ctx.save_for_backward in the forward pass. Thus when we
            # unshard parameters, we should reuse the original Tensor
            # Storage object and unshard it in-place. For now, just resize
            # the Storage to 0 to save memory.
            free_storage_(p._full_param_padded)

    def local_metadata_dict(self) -> Dict[str, Any]:
        """
        Get the information needed to reconstruct the model from shards offline.
        """

        params_metadata = []

        for path, m in self.named_modules():
            if not isinstance(m, FullyShardedDataParallel):
                continue

            # Dealing with FSDP(flatten_parameter=False)
            # There are as many sharded parameters as there parameters in the
            # consolidated model, so we only need to export how to reshape the
            # parameters to their orginal shape and take care of the padding
            if not m.flatten_parameters:
                params_metadata.append(
                    {
                        "fsdp_path": _clean_path(path),
                        "is_flat": False,
                        "num_padded": m.numel_padded_per_param,
                        "param_names": [_clean_path(p) for p in m.param_paths],
                        "param_shapes": [p._orig_size for p in m.params],
                        "param_numels": [_numel_from_size(p._orig_size) for p in m.params],
                        "no_broadcast_optim_state": m.no_broadcast_optim_state,
                    }
                )

            # Dealing with FSDP(flatten_parameter=True)
            # Now, there is just one flattened parameter mapped to N different
            # parameters, so we need to export additional information (numels)
            # on how to split the "merged" parameters, by extracting the meta-data
            # used in the FlattenParamsWrapper
            else:
                param_names = []
                for module_path, param_name in m.param_path_infos:
                    full_param_path = module_path + "." + param_name if module_path else param_name
                    param_names.append(_clean_path(full_param_path))
                params_metadata.append(
                    {
                        "fsdp_path": _clean_path(path),
                        "is_flat": True,
                        "num_padded": m.numel_padded_per_param,
                        "param_names": param_names,
                        # TODO (Min): we don't want to access the private _param_shapes and
                        # _param_numels here. We want to dump metadata from FPW when there are
                        # multiple groups of params.
                        "param_shapes": m._fsdp_wrapped_module.flat_param._param_shapes,  # type: ignore
                        "param_numels": m._fsdp_wrapped_module.flat_param._param_numels,  # type: ignore
                        "no_broadcast_optim_state": m.no_broadcast_optim_state,
                    }
                )

        buffer_names = [_clean_path(buffer_name) for buffer_name, _ in self.named_buffers(recurse=True)]
        return dict(param_metadata=params_metadata, buffer_names=buffer_names)

    @staticmethod
    def consolidate_shard_weights(
        shard_weights: List[Dict[str, torch.Tensor]],
        shard_metadata: List[Dict[str, Any]],
        with_module_buffers: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Given a list of weights and meta data associated to N shards, reconstruct
        the weights of an equivalent consolidated (non-sharded) model.

        Module parameters are consolidated using the shard metadata.

        Module buffers are taken from shard 0: this assumes that module buffers
        are either synchronized or that the shard 0 value is valid for all shards.
        If this behavior is not correct for your module (for instance if buffers
        needs to be reduced instead), you can disable it with `with_module_buffers=False`.

        This method is used to re-assemble checkpoints of shards without
        having to instantiate FSDP wrappers with the world size originally used
        to save the shards.
        """
        if len(shard_weights) != len(shard_metadata) or not len(shard_weights):
            raise ValueError("Require meta data for each shard and non-empty shards")

        consolidated_weights = {}
        original_world_size = len(shard_weights)

        # Deal with the parameters of the model, for which there should be
        # a corresponding entry in the metadata
        shard_0_metadata = shard_metadata[0]["param_metadata"]
        num_fsdp_wrappers = len(shard_0_metadata)
        for fsdp_wrapper_index in range(num_fsdp_wrappers):
            fsdp_path = shard_0_metadata[fsdp_wrapper_index]["fsdp_path"]
            param_names = shard_0_metadata[fsdp_wrapper_index]["param_names"]
            param_numels = shard_0_metadata[fsdp_wrapper_index]["param_numels"]
            param_shapes = shard_0_metadata[fsdp_wrapper_index]["param_shapes"]

            # Dealing with FSDP(flatten_parameter=False)
            # For each parameter of the FSDP wrapper, get rid of the padding on each shard,
            # concatenate the shards and reshape them to their initial shape
            if not shard_0_metadata[fsdp_wrapper_index]["is_flat"]:
                for i in range(len(param_names)):
                    param_name = param_names[i]
                    param_name = ".".join([fsdp_path, param_name]) if fsdp_path else param_name
                    shards = []
                    for rank in range(original_world_size):
                        shard = shard_weights[rank][param_name]
                        pad = shard_metadata[rank]["param_metadata"][fsdp_wrapper_index]["num_padded"][i]
                        shards.append(_unpad(shard, pad))
                    full_flatten_param = torch.cat(shards, dim=0)
                    consolidated_weights[param_name] = full_flatten_param.view(param_shapes[i])

            # Dealing with FSDP(flatten_parameter=True)
            # Concatenate the merged flat_param after removing the padding
            # and then split the flat_param by using numel, before reshaping each
            # split to the original shape
            else:
                # Concatenate the flat_param parameter after removing the padding
                flat_param_name = ".".join([fsdp_path, "flat_param_0"]) if fsdp_path else "flat_param_0"
                shards = []
                for rank in range(original_world_size):
                    shard = shard_weights[rank][flat_param_name]
                    pad = shard_metadata[rank]["param_metadata"][fsdp_wrapper_index]["num_padded"][0]
                    shards.append(_unpad(shard, pad))
                full_flatten_param = torch.cat(shards, dim=0)

                # Split the flat_param into its constituents
                assert sum(param_numels) == full_flatten_param.size(0)
                for n, t, s in zip(param_names, full_flatten_param.split(param_numels), param_shapes):
                    full_name = fsdp_path + "." + n if fsdp_path else n
                    consolidated_weights[full_name] = t.view(s)

        # Deal with the buffers, which are not parameters and are not sharded by FSDP
        # and therefore are replicated among the different shards.
        # We take the values of the first shard (this assumes that there is some form
        # of synchronization between shards or that all shards buffers are equivalent)
        if with_module_buffers:
            for buffer_name in shard_metadata[0]["buffer_names"]:
                consolidated_weights[buffer_name] = shard_weights[0][buffer_name]

        return consolidated_weights

    @torch.no_grad()
    def _use_fp32_param_shard(self, params: Optional[List[Parameter]] = None) -> None:
        """Use FP32 shard for a list of params."""
        if params is None:
            params = self.params
        for p in params:
            p.data = p._fp32_shard

    @torch.no_grad()
    def _cast_fp32_param_shards_to_fp16(self, params: Optional[List[Parameter]] = None) -> None:
        """Cast FP32 param shard to FP16 for a list of params."""
        if params is None:
            params = self.params
        with torch.cuda.stream(self._streams["fp32_to_fp16"]):
            for p in params:
                assert p._fp16_shard is not None
                alloc_storage_(p._fp16_shard, size=p._fp32_shard.size())
                p._fp16_shard.copy_(
                    # If cpu_offload is True, this will be non-blocking because
                    # _fp32_shard is pinned, otherwise it's a no-op.
                    p._fp32_shard.to(p._fp16_shard.device, non_blocking=True)
                )
                p.data = p._fp16_shard
        torch.cuda.current_stream().wait_stream(self._streams["fp32_to_fp16"])

    @torch.no_grad()
    def _free_fp16_param_shard(self, params: Optional[List[Parameter]] = None) -> None:
        """Free storage for FP16 shards for a list of params."""
        if params is None:
            params = self.params
        current_stream = torch.cuda.current_stream()
        for p in params:
            if p._fp16_shard is not None:
                # _fp16_shard is allocated in "fp32_to_fp16" stream, so we can't
                # free it until the work in the current stream completes.
                p._fp16_shard.record_stream(current_stream)
                free_storage_(p._fp16_shard)

    def assert_state(self, state: Union[TrainingState, List[TrainingState]]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState):
            state = [state]
        if self.training_state not in state:
            msg = f"expected to be in states {state} but current state " f"is {self.training_state}"
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    def _broadcast_pad_info_to_r0(self) -> List[List[List[int]]]:
        """Collect [x.numel_padded_per_param for x in self._fsdp_instances] from teach rank."""
        dummy_tensor = torch.tensor([0], dtype=torch.uint8, device=self.compute_device)
        world_pad_info: List[List[List[int]]] = []  # this will contain values from the whole world.
        for rank in range(self.world_size):
            if rank == self.rank:
                pad_info = [m.numel_padded_per_param for m in self._fsdp_instances]
            else:
                pad_info = dummy_tensor  # type: ignore
            pad_info = broadcast_object(
                pad_info, src_rank=rank, group=self.process_group, dist_device=self.compute_device
            )
            if self.rank == 0:
                world_pad_info.append(pad_info)  # type: ignore
        return world_pad_info

    def _gather_optim_state(
        self, sd_state: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[int, Dict[str, List]], Dict[int, Dict[str, List]]]:
        """For each value in state[i], if the value is a tensor, collect it from the world. Else use rank 0's entry."""
        gathered_state: Dict[int, Dict[str, List[Any]]] = {}
        singleton_state: Dict[int, Dict[str, List[Any]]] = {}  # Dimensionless tensor
        for k, v in sd_state.items():
            gathered_state[k] = {}
            singleton_state[k] = {}
            desired_buffer_size = self._fsdp_instances[k].flat_param._full_param_padded.size()  # type: ignore
            buffer = None  # for sharded tensors
            singleton_buffer = None  # for singleton tensors
            for buffer_name, t in v.items():
                if torch.is_tensor(t):
                    t = t.to(self.compute_device)

                if ou.is_singleton_tensor(t):
                    if singleton_buffer is None:
                        singleton_buffer = list(t.new_zeros(self.world_size).chunk(self.world_size))
                    dist.all_gather(singleton_buffer, t, group=self.process_group)
                    if self.rank == 0:
                        singleton_state[k][buffer_name] = [x.cpu().squeeze() for x in singleton_buffer]
                        assert ou.is_singleton_tensor(singleton_state[k][buffer_name][0])
                elif torch.is_tensor(t):
                    if buffer is None:
                        buffer = list(t.new_zeros(*desired_buffer_size).chunk(self.world_size))
                    dist.all_gather(buffer, t, group=self.process_group)
                    if self.rank == 0:
                        gathered_state[k][buffer_name] = [x.cpu() for x in buffer]
                elif self.rank == 0:  # Add non tensor state
                    gathered_state[k][buffer_name] = [t]

        return gathered_state, singleton_state

    def gather_full_optim_state_dict(self, optim: torch.optim.Optimizer, **ignored: Dict) -> Optional[Dict[str, Any]]:
        """Return the last known global optimizer state. The returned state is compatible with Pytorch, in that the
        sharded properties are not exposed. Multiple parameter groups are not yet supported.

        This should be called only on the root FSDP instance.
        Nested FSDP instances are supported as long as they have the same world_size as the parent or world_size=1.

        Args:
            optim (Optimizer): an optimizer instance for this FSDP rank. Its state_dict is
                        used in the consolidation. However, its state is not modified.

        Returns:

            * A dict with four entries (On rank zero, other workers return ``None``)
                * state - a dict holding gathered optimization state, 1 entry per unflat parameter
                * param_groups - a dict containing the 1 parameter group
                * param_id_map - global (unflat) to local (flat) id mapping
                * uncollected_local_ids - keys in the state dict that were not broadcast

        """
        if not self.flatten_parameters:
            raise NotImplementedError("optim state dict requires flatten_parameters=True")

        self._lazy_init()
        sd = self._remove_uncollectable_params_from_optim_state_dict(optim.state_dict())
        assert set(sd.keys()) == {"param_groups", "state"}, f'{set(sd.keys())} != {"param_groups", "state"}'
        assert len(sd["param_groups"]) == 1, "Param groups are not supported"
        # We use all_gather to consolidate OSD['state'] and broadcast to consolidate the other keys (like param_groups)
        state, singleton_state = self._gather_optim_state(sd.pop("state"))
        pad_info = self._broadcast_pad_info_to_r0()
        if self.rank != 0:
            return None
        # Unify the shard states by concatenating tensors and unflattening params
        new_state_dict = ou.build_unflat_state_dict(
            self._fsdp_instances, pad_info, state, singleton_state, self.uncollected_opt_state, sd["param_groups"]
        )
        self.uncollected_opt_state = {}
        assert "uncollected_local_ids" in new_state_dict
        return new_state_dict

    @property
    def _fsdp_instances(self) -> List[nn.Module]:
        """Returns all fsdp modules in self.modules() including self."""
        return [m for m in self.modules() if isinstance(m, FullyShardedDataParallel)]

    def _remove_uncollectable_params_from_optim_state_dict(self, osd: Dict) -> Dict:
        uncollected_ids = [i for i, m in enumerate(self._fsdp_instances) if m.no_broadcast_optim_state]
        new_dct = {"state": {k: v for k, v in osd["state"].items() if k not in uncollected_ids}}
        if self.rank == 0:
            # Save placeholders for uncollected opt state to keep the same unflat OSD format, and move them to CPU.
            self.uncollected_opt_state = {
                k: recursive_copy_to_device(v, non_blocking=False, device=torch.device("cpu"))
                for k, v in osd["state"].items()
                if k in uncollected_ids
            }

        pg = copy.deepcopy(osd["param_groups"])
        new_dct["param_groups"] = pg
        return new_dct

    def get_shard_from_optim_state_dict(self, full_optim_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get the portion of the optimizer state dict associated with the shard

        This can be used to get the right sharded optimizer state to be loaded
        into the sharded optimizer for this FSDP rank.

        Args:
            full_optim_state_dict (dict): consolidated optimizer state returned by ``gather_full_optim_state``, or loaded from a checkpoint.

        Returns:
            (dict): a shard of the optimizer state.
        """
        # Assert nesting is the same as it was at save time
        instance_list = self._fsdp_instances
        ou.check_param_counts_before_sharding(full_optim_state_dict, len(instance_list))
        ids_not_to_shard = copy.deepcopy(full_optim_state_dict["uncollected_local_ids"])
        if self.flatten_parameters:
            full_optim_state_dict = ou.flatten_optim_state_dict(full_optim_state_dict)
            assert len(full_optim_state_dict["state"]) in (
                0,
                len(instance_list),
            ), f'{len(full_optim_state_dict["state"])}, {len(instance_list)}'

        # get the portion of dict associated with the shard, in place
        for id, s in full_optim_state_dict["state"].items():
            for k, v in s.items():
                if torch.is_tensor(v) and id not in ids_not_to_shard:
                    v_shard, _ = self._get_shard(v)
                elif isinstance(v, list) and ou.is_singleton_tensor(v[0]):
                    # if we are resuming on larger world size, take first entry
                    v_shard = v[0] if self.rank >= len(v) else v[self.rank]
                    assert ou.is_singleton_tensor(v_shard)
                else:
                    v_shard = v  # don't shard entries that are not tensors
                full_optim_state_dict["state"][id][k] = v_shard

        return full_optim_state_dict

    def _print_r0(self, msg: str, restart: bool = False) -> None:
        """Debugging utility to print memory usage stats nicely on rank 0"""
        if restart:
            self._tstart = time.time()
        if self.rank == 0:
            gb_denom = 1024 ** 3
            logging.info(
                f"{msg} cur={torch.cuda.memory_allocated()/gb_denom: .4f} GB, max={torch.cuda.max_memory_allocated()/gb_denom: .4f} GB, t={time.time()-self._tstart: .1f}"
            )

    # Note: This property will be deprecated in an upcoming release in favor of `move_params_to_cpu`.
    @property
    def cpu_offload(self) -> bool:
        return self.move_params_to_cpu


def _get_default_cuda_device(module: nn.Module) -> torch.device:
    """Try to infer CUDA device from module parameters."""
    try:
        compute_device = next(module.parameters()).device
        if compute_device.type == "cuda":
            return compute_device
    except StopIteration:
        pass
    # Fall back to current CUDA device
    return torch.device("cuda")


def cast_floats_to_right_precision(to_fp16: bool, no_grad: bool, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """
    Cast floating point Tensors in *args or **kwargs to FP16 or FP32 if they are not.
    We also retain the requires_grad flag so that casting doesn't affect the autograd graph.
    """

    def fn_fp16(x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.float32:
            y = x.half()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x

    def fn_fp32(x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.float16:
            y = x.float()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x

    fn = fn_fp16 if to_fp16 else fn_fp32
    context = torch.no_grad() if no_grad else contextlib.suppress()
    with context:  # type: ignore
        return apply_to_tensors(fn, args), apply_to_tensors(fn, kwargs)


def free_storage_(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


@torch.no_grad()
def alloc_storage_(data: torch.Tensor, size: torch.Size) -> None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():  # no need to reallocate
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())


def _post_state_dict_hook(
    module: FullyShardedDataParallel, state_dict: "OrderedDict[str, torch.Tensor]", prefix: str, *args: Any
) -> "OrderedDict[str, torch.Tensor]":
    # Assuming we are in a ``summon_full_params()`` context, we need to clone
    # each tensor so that it does not get freed (in-place) when the context
    # exits. At the same time, this hook can be called multiple times
    # recursively, so we need to make sure that we only clone each tensor at
    # mostonce. Thus we add an attribute on the tensor called "_has_been_cloned"
    # which keeps track of tensors that are no longer at risk of being freed.
    for key in state_dict.keys():
        if not key.startswith(prefix) or getattr(state_dict[key], "_has_been_cloned", False):
            continue
        if state_dict[key].device.type != module.state_dict_device.type:
            state_dict[key] = state_dict[key].to(device=module.state_dict_device)
            state_dict[key]._has_been_cloned = True
        elif module.training_state == TrainingState.SUMMON_FULL_PARAMS:
            # We copy the state_dict since full param will be freed after we
            # exit the ``summon_full_params()`` context.
            state_dict[key] = state_dict[key].clone()
            state_dict[key]._has_been_cloned = True

    # Remove "_fsdp_wrapped_module." prefix
    replace_by_prefix_(state_dict, prefix + "_fsdp_wrapped_module.", prefix)
    return state_dict


def _pre_load_state_dict_hook(
    state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], prefix: str, *args: Any
) -> None:
    replace_by_prefix_(state_dict, prefix, prefix + "_fsdp_wrapped_module.")


def _clean_path(path: str) -> str:
    return ".".join([split for split in path.split(".") if split not in {"_fsdp_wrapped_module", "_fpw_module"}])


def _numel_from_size(size: torch.Size) -> int:
    numel = 1
    for dim in size:
        numel *= dim
    return numel


def _unpad(shard: torch.Tensor, pad: int) -> torch.Tensor:
    if pad > 0:
        shard = shard[:-pad]
    return shard


########################################################################################
# Below are APIs used together with FSDP, but not directly part of FSDP.
########################################################################################


def auto_wrap_bn(
    module: nn.Module,
    single_rank_pg: bool = False,
    process_group: Optional[ProcessGroup] = None,
    fsdp_config: Optional[Dict[str, Any]] = None,
    wrap_it: bool = True,
    assert_on_collision: bool = True,
) -> nn.Module:
    """
    Auto wrap all BatchNorm (BN) instances with a safer FSDP, esp. when convert
    to sync BN is used and the outer FSDP is flattening.

    We put BN in is own full precision, unflatten, single GPU group FSDP.  Note, SyncBNs still have
    a group size == world_size. The input and output for BN are still FP16 in mixed precision mode.
    See ``keep_batchnorm_fp32`` here: https://nvidia.github.io/apex/amp.html

    This needs to be done at each rank, like models being wrapped by FSDP at each rank.

    Args:
        module (nn.Module):
            The model (or part of the model) in which BN to be pre-wrapped.
        single_rank_pg (bool):
            If true, put BNs in a single-rank process group. Default False.
            This might be needed for Apex sync BN support. Still under construction.
        process_group (ProcessGroup):
            Optional process group to be used.
        fsdp_config (Dict):
            Optional fsdp_config to be used.
        wrap_it (bool):
            Whether or not wrap the module after setting the config.
            Default: True
        assert_on_collision (bool):
            Whether or not assert if a wrapper_config already exists on the module.
            Default: True

    Returns:
        Processed module, where BNs are wrapped with a special FSDP instance.
    """
    # Prepare a fsdp_config dict for BNs.
    pg = process_group
    if single_rank_pg:
        # No sharding with this single member group.
        my_rank = dist.get_rank()
        pg = get_process_group_cached(ranks=[my_rank])

    if fsdp_config is None:
        fsdp_config = {
            "process_group": pg,
            "mixed_precision": False,  # Keep the weights in FP32.
            "flatten_parameters": False,  # Do not flatten.
            # Reshard==False is good for performance. When FSDP(checkpoint(FSDP(bn))) is used, this
            # **must** be False because BN's FSDP wrapper's pre-backward callback isn't called
            # within the checkpoint's outer backward when multiple forward passes are used.
            "reshard_after_forward": False,
            # No bucketing or small bucketing should be enough for BNs.
            "bucket_cap_mb": 0,
            # Setting this for SyncBatchNorm. This may have a performance impact. If
            # SyncBatchNorm is used, this can be enabled by passing in the `fsdp_config` argument.
            "force_input_to_fp32": False,
        }

    # Assign the config dict to BNs.
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if assert_on_collision:
                assert not hasattr(
                    m, "wrapper_config"
                ), "Module shouldn't already have a wrapper_config. Is it tagged already by another policy?"
            m.wrapper_config = fsdp_config

    # Wrap it.
    with (
        enable_wrap(config_auto_wrap_policy, wrapper_cls=FullyShardedDataParallel) if wrap_it else contextlib.suppress()
    ):
        return auto_wrap(module)
