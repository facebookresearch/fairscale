# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from enum import Enum, auto
import functools
from typing import TYPE_CHECKING, Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairscale.nn.misc import FlattenParamsWrapper
from fairscale.utils.containers import (
    apply_to_tensors,
    pack_kwargs,
    split_non_tensors,
    unpack_kwargs,
    unpack_non_tensors,
)
from fairscale.utils.parallel import compute_shard_size, validate_process_group

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class TrainingState(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.

    TODO (Min): It would be nice to capture the stepping state as well.
        Maybe we can use the model.zero_grad() call, but not sure if it
        is called if optim.zero_grad() is used instead.
        It would be nice to have clear state transition be explicit like:

        zero_grad -> fwd -> bwd -> optionally accum grad by repeating
        fwd/bwd -> stepping -> loop back to zero_grad
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD = auto()


class FullyShardedDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters. This is inspired by `Xu et al.`_
    as well as the ZeRO Stage 3 from the DeepSpeed_ work.

    .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
    .. _DeepSpeed: https://www.deepspeed.ai/

    Usage::

        sharded_module = FullyShardedDataParallel(my_module)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        x = sharded_module(x, y=3, z=torch.Tensor([1]))
        loss = x.sum()
        loss.backward()
        optim.step()

    It is also possible to shard individual layers separately and have an outer
    wrapper handle any leftover parameters. This can be helpful to further
    reduce memory usage and to improve training speed by distributing the
    unsharding (all-gather) across the forward pass. For example::

        sharded_model = FullyShardedDataParallel(
            nn.Sequential(
                nn.Linear(5, 100),
                FullyShardedDataParallel(nn.Linear(100, 100)),
                FullyShardedDataParallel(nn.Linear(100, 100)),
                nn.Linear(100, 5),
            )
        )

    Args:
        module (nn.Module): module to checkpoint
        process_group (Optional): process group for sharding
        reshard_after_forward (bool, Optional): if ``True``, reshard parameters
            after the forward pass. This saves memory but slows training. This
            is only relevant when resharding individual layers.
        mixed_precision (bool, Optional): if ``True``, inputs, activations and
            gradients will be kept in FP16; computation and communication will
            occur in FP16; and a (sharded) master copy of the model weights will
            be maintained in FP32.
        fp32_reduce_scatter (bool, Optional): if ``True``, then reduce-scatter
            gradients in FP32. This is only relevant when *``mixed_precision``*
            is ``True``.
        flatten_parameters (bool, Optional): if ``True``, flatten parameters
            into a single contiguous tensor, which improves training speed.
        cpu_offload (bool, Optional): if ``True``, offload FP32 params to CPU.
            This is only relevant when *``mixed_precision``* is ``True``.
        compute_dtype (torch.dtype, Optional): dtype for full parameters for
            computation. This defaults to ``torch.float32`` unless
            *``mixed_precision``* is set, in which case it defaults to
            ``torch.float16``.
        move_grads_to_cpu (bool, Optional): move gradient shard to CPU after
            reduction. This is useful when combined with CPU-based optimizers.
            It defaults to the value of *``cpu_offload``*.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        reshard_after_forward: bool = True,
        mixed_precision: bool = False,
        fp32_reduce_scatter: bool = False,
        flatten_parameters: bool = True,
        cpu_offload: bool = False,
        compute_dtype: Optional[torch.dtype] = None,
        move_grads_to_cpu: Optional[bool] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.new_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        self.reshard_after_forward = reshard_after_forward
        self.mixed_precision = mixed_precision
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self.flatten_parameters = flatten_parameters
        self.cpu_offload = cpu_offload
        self.compute_dtype = compute_dtype or (torch.float16 if mixed_precision else torch.float32)
        self.move_grads_to_cpu = cpu_offload if move_grads_to_cpu is None else move_grads_to_cpu

        if self.fp32_reduce_scatter and not self.mixed_precision:
            raise ValueError("fp32_reduce_scatter requires mixed_precision=True")
        if self.cpu_offload and not self.mixed_precision:
            raise ValueError("cpu_offload requires mixed_precision=True")

        compute_device = torch.device("cuda") if self.cpu_offload else next(module.parameters()).device
        validate_process_group(compute_device, self.process_group)

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        params = list(p for p in module.parameters() if not getattr(p, "_is_sharded", False))

        if self.flatten_parameters and len(params) > 0:
            self.module: nn.Module = FlattenParamsWrapper(module, param_list=params)
            del module  # free original module in case it helps garbage collection
            self.params = [self.module.flat_param]
        else:
            self.module = module
            self.params = params

        # Shard module parameters in place
        self._shard_parameters_()

        # Make sure all parameters are sharded.
        for n, p in self.named_parameters():
            assert getattr(p, "_is_sharded", False), f"found unsharded parameter: {n} ; {p.size()}"

        self._reset_lazy_init()

        # Flag to indicate if we require gradient reduction in the backward
        # pass. This will be False when inside the no_sync context manager.
        self.require_backward_grad_sync: bool = True

        self.training_state = TrainingState.IDLE

    @torch.no_grad()
    def _all_buffers_to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        """Move all buffers to the specified device and dtype, recursively."""
        cast_fn = functools.partial(cast_buffers_, device=device, dtype=dtype)
        self.apply(cast_fn)

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
        for p in self.params:
            assert not hasattr(p, "_is_sharded")
            assert p.is_floating_point()
            if self.mixed_precision:
                assert p.dtype == torch.float32

            p._is_sharded = True
            p._orig_size = p.data.size()

            # shard p.data such that all elements are part of a shard and the
            # last shard is <= all other shards or, all shards are 1-element
            # in size in case total size is smaller than the world_size.
            #
            # This way, we don't have holes when shards are reconstructed and
            # only extra padding elements need to added/removed during sharding.
            shard_size = compute_shard_size(p.data.numel(), self.world_size)
            s = min(self.rank * shard_size, p.data.numel())
            e = min(s + shard_size, p.data.numel())
            assert (
                0 <= s <= e <= p.data.numel()
            ), f"_shard_parameters_: {p.data.numel()} {self.world_size} {shard_size} {s} {e}"

            orig_data = p.data
            p.data = torch.flatten(p.data)[s:e].clone()
            if p.data.numel() < shard_size:
                p.data = F.pad(p.data, [0, shard_size - p.data.numel()])  # pad zeros to the right size.
            assert p.data.numel() == shard_size, f"{p.data.numel()} {shard_size}"
            free_storage_(orig_data)

    @torch.no_grad()
    def _all_gather_full_param(self, p: nn.Parameter) -> None:
        """Fill p._full_param with gathered p.data values (using torch.distributed.all_gather).

        The p._full_param is already allocated and have the size equal
        to shard_size * world_size.

        It is up to the caller to do necessary resize/reshape to the
        unpadded _full_param.
        """
        full_param_chunks = list(p._full_param.chunk(self.world_size))
        assert len(full_param_chunks) == self.world_size
        assert full_param_chunks[-1].numel() == p.data.numel(), f"{full_param_chunks[-1].numel()} {p.data.numel()}"
        param_shard = p.data  # we will gather this from each worker

        dist.all_gather(full_param_chunks, param_shard, group=self.process_group)
        # ^ updates p._full_param

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
        state["orig_sizes"] = [p._orig_size for p in self.params]
        if state["process_group"] is not None:
            state["process_group"] = "MISSING"  # process_group isn't pickleable
        self._reset_lazy_init()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Intercept state setting and perform needed changes on params."""
        super().__setstate__(state)

        def fixup(p: Parameter, size: torch.Size) -> Parameter:
            assert isinstance(p, Parameter)
            p.data = p.data.clone()  # move tensors out of shared memory
            p._is_sharded = True
            p._orig_size = size
            return p

        self.params = [fixup(p, size) for p, size in zip(self.params, self.orig_sizes)]
        del self.orig_sizes
        self._reset_lazy_init()

    # TODO (Min): figuring out how to do typing for this overloaded function.
    def state_dict(self, *args, **kwargs):  # type: ignore
        """
        Returns the whole (unsharded) state of the module. Parameters are not
        sharded, so the resulting state_dict can be loaded directly by the
        wrapped Module without any sharding-specific logic. Returned tensors will always be typed float32
        """
        torch.cuda.synchronize()
        self._lazy_init()
        self._rebuild_full_params()
        self._all_buffers_to(dtype=torch.float32)  # Buffers dtype stays consistent with parameters.
        state_dict = self.module.state_dict(*args, **kwargs)
        # We don't free the params after generating the state dict, since
        # freeing is done in-place (via the Storage) and would corrupt the
        # returned state dict. However, we need to maintain the invariant that
        # p.data corresponds to the FP32 param shard, so we do that here.
        self._use_fp32_param_shard()
        self._all_buffers_to(dtype=self.compute_dtype)
        return state_dict

    # TODO (Min): figuring out how to do typing for this overloaded function.
    def local_state_dict(self, *args, **kwargs):  # type: ignore
        """
        Returns the local (sharded) state of the module. Parameters are sharded,
        so the resulting state_dict can only be loaded after the Module has been
        wrapped with FullyShardedDataParallel.
        """
        if self.flatten_parameters:
            return self.module.flat_state_dict(*args, **kwargs)  # type: ignore
        else:
            return self.module.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """Load a whole (unsharded) state_dict."""
        torch.cuda.synchronize()
        self._lazy_init()
        self._rebuild_full_params()
        output = self.module.load_state_dict(state_dict, strict)
        self._free_full_params()
        return output

    def load_local_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """Load a local (sharded) state_dict."""
        return self.module.load_state_dict(state_dict, strict)

    @contextlib.contextmanager
    def no_sync(self) -> Generator:
        """
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.
        """
        assert self._is_root, "no_sync on inner FSDP is not tested."
        self.assert_idle()
        # This instance may wrap other FullyShardedDataParallel instances and we
        # need to set all of them to accumulate gradients.
        old_flags = []
        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel):
                old_flags.append((m, m.require_backward_grad_sync))
                m.require_backward_grad_sync = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                m.require_backward_grad_sync = old_flag

    def _reset_lazy_init(self) -> None:
        """Reset instance so :func:`_lazy_init` will run on the next forward."""
        self._is_root: Optional[bool] = None
        self._streams: Dict[str, torch.cuda.Stream] = {}

    def _lazy_init(self) -> None:
        """Initialization steps that should happen lazily, typically right
        before the first forward pass."""
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
        if self.cpu_offload:  # Buffers stay on GPU, and dont get sharded
            self._all_buffers_to(device=torch.device("cuda"), dtype=self.compute_dtype)
        else:
            self._all_buffers_to(dtype=self.compute_dtype)

        # Don't free the full params for the outer-most (root) instance, since
        # those params will be needed immediately after for the backward pass.
        if self._is_root:
            self.reshard_after_forward = False

    @torch.no_grad()
    def _init_param_attributes(self, p: Parameter) -> None:
        """
        We manage several attributes on each Parameter instance. The first two
        are set by :func:`_shard_parameters_`:

            ``_is_sharded``: ``True`` after the Parameter is initially sharded
            ``_orig_size``: the size of the original Parameter (before sharding)

        The remaining attributes are set here:
            ``_fp32_shard``: a single shard of the parameters in full precision
                (typically FP32, but this is dependent on the dtype of the model
                as it's passed in by the user). This can be on CPU or GPU
                depending on the value of *``cpu_offload``*.
            ``_fp16_shard``: if *``mixed_precision``* is ``True``, this will be
                a single shard of the parameters in FP16, used for all-gather.
            ``_full_param``: the full weight, used for computation in the
                forward/backward pass. This will be resized in place and only
                materialized (via all-gather) as needed.
        """
        assert p._is_sharded and hasattr(p, "_orig_size")
        if hasattr(p, "_full_param"):
            return

        # Compute device defaults to CUDA when *cpu_offload* is enabled, or the
        # param's current device otherwise (could be CPU).
        compute_device = torch.device("cuda") if self.cpu_offload else p.device

        # A single shard of the parameters in full precision.
        p._fp32_shard = p.data

        if self.mixed_precision:
            assert p._fp32_shard.dtype == torch.float32

            if self.cpu_offload:
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
            p._fp16_shard = torch.zeros_like(p._fp32_shard, device=compute_device, dtype=self.compute_dtype)
            free_storage_(p._fp16_shard)
        else:
            p._fp16_shard = None  # use _fp32_shard

        # We also maintain a full-sized parameter of type self.compute_dtype
        # (FP16 for mixed_precision or FP32 otherwise). We resize the
        # storage to size 0 at init (here) and only materialize as needed.
        p._full_param = torch.zeros(p._orig_size, device=compute_device, dtype=self.compute_dtype)
        free_storage_(p._full_param)

        if self.move_grads_to_cpu:
            # We can optionally move the grad shard to CPU during the backward
            # pass. In this case, it's important to pre-allocate the CPU grad
            # shard in pinned memory so that we can do a non-blocking transfer.
            p._cpu_grad = torch.zeros_like(p.data, device="cpu").pin_memory()

    def _set_is_root(self) -> None:
        """If ``True``, implies that no other :class:`FullyShardedDataParallel`
        instance wraps this one. Called once by :func:`_lazy_init`."""
        if self._is_root is not None:
            return
        # No FullyShardedDataParallel instance wraps this, else _is_root would be set to False
        self._is_root = True
        # As the root, we now set all children instances to False.
        for n, m in self.named_modules():
            if n != "" and isinstance(m, FullyShardedDataParallel):
                assert m._is_root is None
                m._is_root = False

    def _setup_streams(self) -> None:
        """Create streams to overlap data transfer and computation."""
        if len(self._streams) > 0 or not self._is_root:
            return
        # Stream to move main FP32 params (may be on CPU) to FP16 for forward.
        self._streams["fp32_to_fp16"] = torch.cuda.Stream()
        # Stream for all-gathering parameters.
        self._streams["all_gather"] = torch.cuda.Stream()
        # Stream for overlapping grad reduction with the backward pass.
        self._streams["post_backward"] = torch.cuda.Stream()
        # We share streams with all children instances, which allows them to
        # overlap transfers across the forward pass without synchronizing with
        # the default stream.
        for n, m in self.named_modules():
            if n != "" and isinstance(m, FullyShardedDataParallel):
                m._streams = self._streams

    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`FullyShardedDataParallel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if self.mixed_precision:
            self._streams["fp32_to_fp16"].wait_stream(torch.cuda.current_stream())
        else:
            self._streams["all_gather"].wait_stream(torch.cuda.current_stream())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._lazy_init()

        # Start of a forward pass.
        self.training_state = TrainingState.FORWARD

        # Due to the use of streams, we need to make sure the previous
        # ``optim.step()`` is done before we all-gather parameters.
        if self._is_root:
            self._wait_for_previous_optim_step()

        if self.mixed_precision:
            args, kwargs = cast_inputs_to_fp16(*args, **kwargs)

        # All-gather full parameters. This will also transfer FP32 parameters to
        # ``self.compute_dtype`` (e.g., FP16 if *mixed_precision* is ``True``).
        self._rebuild_full_params()

        # Register backward hooks to reshard params and reduce-scatter grads.
        # These need to be re-registered every forward pass.
        self._register_post_backward_hooks()

        outputs = self.module(*args, **kwargs)

        if self.reshard_after_forward:
            self._free_full_params()

        # Switch to main FP32 param shard. We maintain this invariant throughout
        # the code, i.e., ``p.data == p._fp32_shard`` after each function. This
        # also ensures that after the first forward, the optimizer state will be
        # initialized with the correct dtype and (sharded) size, since optimizer
        # state is typically initialized lazily in ``optim.step()``.
        self._use_fp32_param_shard()

        # Register pre-backward hooks to all-gather the params for the backward
        # pass (if needed).
        outputs = self._register_pre_backward_hooks(outputs)

        # Done with a forward pass.
        self.training_state = TrainingState.IDLE

        return outputs

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:
        """Register pre-backward hook to run before the wrapped module's
        backward. Hooks should be attached to all outputs from the forward."""
        if not torch.is_grad_enabled():
            return outputs  # don't register hooks if grad isn't enabled

        pre_backward_hook_has_run = [False]

        def _pre_backward_hook(*unused: Any) -> None:
            if pre_backward_hook_has_run[0]:
                return  # only run once
            pre_backward_hook_has_run[0] = True

            # Start of a backward pass.
            self.training_state = TrainingState.BACKWARD

            # All-gather full parameters.
            if self.reshard_after_forward:
                self._rebuild_full_params()
            else:
                self._use_full_params()
            # Make sure p.grad has the correct size/device (or set it to None).
            self._prep_grads_for_backward()

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            t.register_hook(_pre_backward_hook)
            return t

        # Attach hooks to Tensor outputs.
        outputs = apply_to_tensors(_register_hook, outputs)

        return outputs

    def _register_post_backward_hooks(self) -> None:
        """Register backward hooks to reshard params and reduce-scatter grads."""
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        self._post_backward_callback_queued = False
        for p in self.params:
            if p.requires_grad:
                if hasattr(p, "_shard_bwd_hook"):
                    p._shard_bwd_hook[1].remove()  # remove existing handle
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p))
                p._shard_bwd_hook = (grad_acc, handle)

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
        """
        At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will replace
        ``param.grad`` with a single shard of the summed gradient across all
        GPUs.  This shard will align with the current GPU rank. For example::

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
        if param.grad is None:
            return
        if param.grad.requires_grad:
            raise RuntimeError("FullyShardedDataParallel only works with gradients that don't require grad")

        # Free full params and switch to FP32 shard after backward.
        self._free_full_params([param])
        self._use_fp32_param_shard([param])
        if self.mixed_precision:
            # This is a no-op if reshard_after_forward is True, since we already
            # free the param shard when rebuilding the full params in the
            # pre_backward_hook.
            self._free_fp16_param_shard([param])

        if not self.require_backward_grad_sync:
            return

        # Wait for all work in the current stream to finish, then start the
        # reductions in post_backward stream.
        self._streams["post_backward"].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._streams["post_backward"]):
            if self.mixed_precision and self.fp32_reduce_scatter:
                # Cast grad to FP32.
                param.grad.data = param.grad.data.to(param.dtype)

            if self.world_size > 1:
                # Average grad by world_size for consistency with PyTorch DDP.
                param.grad.data.div_(self.world_size)

                # Reduce-scatter grad.
                param.grad.data = self._reduce_scatter(param.grad.data, param.data.numel())
            else:
                param.grad.data = torch.flatten(param.grad.data)

            # Cast grad to param's dtype (typically FP32). Note: we do this
            # before the move_grads_to_cpu step so that this entire hook remains
            # non-blocking. The downside is a bit more D2H transfer in that case.
            if self.mixed_precision:
                param.grad.data = param.grad.data.to(dtype=param.data.dtype)

            # Optionally move gradients to CPU, typically used if one is running
            # the optimizer on the CPU.
            if self.move_grads_to_cpu:
                param._cpu_grad.copy_(param.grad.data, non_blocking=True)
                param.grad.data = param._cpu_grad

        # Enqueue a callback at the end of the backward pass to ensure that all
        # post-backward work has finished. We only need one callback and it only
        # needs to be called from the outer-most (root) instance.
        if self._is_root and not self._post_backward_callback_queued:
            self._post_backward_callback_queued = True
            Variable._execution_engine.queue_callback(self._wait_for_post_backward)

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward work to finish. Only called on root instance."""
        assert self._is_root
        torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
        if self.move_grads_to_cpu:
            # Wait for the non-blocking GPU -> CPU grad transfers to finish.
            torch.cuda.current_stream().synchronize()
        # A backward pass is done.
        self.training_state = TrainingState.IDLE

    @torch.no_grad()
    def _rebuild_full_params(self) -> None:
        """Gather all shards of params."""
        with torch.cuda.stream(self._streams["all_gather"]):
            if self.mixed_precision:
                self._cast_fp32_param_shards_to_fp16()

            for p in self.params:
                p_size = p.data.numel() * self.world_size
                if p._full_param.storage().size() != p_size:
                    # Allocate based on full size from all shards.
                    p._full_param.resize_(p_size)
                    alloc_storage_(p._full_param, torch.Size((p_size,)))
                    if self.world_size > 1:
                        # Fill p._full_param with (p.data for each shard in self.world_size)
                        self._all_gather_full_param(p)
                        if p._orig_size.numel() < p._full_param.numel():
                            # We need a smaller view into _full_param and save
                            # _full_param_padded.
                            p._full_param_padded = p._full_param
                            # Note, full size can be >> orig_size when world_size is
                            # large and param size is tiny.
                            p._full_param = p._full_param.split(p._orig_size.numel())[0]
                    else:
                        torch.flatten(p._full_param).copy_(p.data)
                    p._full_param = p._full_param.reshape(p._orig_size)

                p.data = p._full_param

                if self.mixed_precision:
                    self._free_fp16_param_shard([p])
        torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

    @torch.no_grad()
    def _use_full_params(self) -> None:
        for p in self.params:
            assert p._full_param.storage().size() != 0
            p.data = p._full_param

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
        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._streams["all_gather"]):
            for p in params:
                # There may be external references to the Tensor Storage that we
                # can't modify, such as references that are created by
                # ctx.save_for_backward in the forward pass. Thus when we
                # unshard parameters, we should reuse the original Tensor
                # Storage object and unshard it in-place. For now, just resize
                # the Storage to 0 to save memory.
                p._full_param.record_stream(current_stream)
                if hasattr(p, "_full_param_padded"):
                    free_storage_(p._full_param_padded)
                    delattr(p, "_full_param_padded")
                else:
                    free_storage_(p._full_param)

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
                # _fp16_shard is allocated in _fp32_to_fp16_stream, so we can't
                # free it until the work in the current stream completes.
                p._fp16_shard.record_stream(current_stream)
                free_storage_(p._fp16_shard)

    @torch.no_grad()
    def _reduce_scatter(self, tensor: torch.Tensor, shard_size: int) -> torch.Tensor:
        """Reduce-scatter a Tensor (gradient from the local worker) and return
        the result (a single "flattened" shard of the summed gradient across workers).

        Shard_size is passed in to compute the padding, but we don't use F.pad since
        it reallocates the tensor, which can be a big chunk of memory consumed. Instead,
        we allocate only for missing and incomplete shards and copy only the needed
        data to the first allocated shard. (The remining allocated shards are just
        padding for reduce_scatter.)
        """
        tensor = torch.flatten(tensor)
        full_shards, rem = divmod(tensor.numel(), shard_size)
        assert full_shards <= self.world_size, (
            f"incorrect shard_size {shard_size} " f"full_shards {full_shards} " f"world_size {self.world_size}"
        )
        full_shards_view = tensor
        if rem > 0:
            # Get two views in to the tensor.
            full_shards_view, rem_view = tensor.split(full_shards * shard_size)

        # This is first part of to_scatter list.
        to_scatter = list(full_shards_view.view(-1, shard_size).unbind(0))

        tail = []
        if full_shards < self.world_size:
            # This is the second part of the to_scatter list.
            tail = [torch.zeros_like(to_scatter[0]) for i in range(full_shards, self.world_size)]

        if rem > 0:
            # Copy the right data in to the first partial shard.
            tail[0][:rem].copy_(rem_view)

        assert len(to_scatter) + len(tail) == self.world_size, (
            f"incorrect length {len(to_scatter)} + {len(tail)} vs. " f"{self.world_size}"
        )

        output = torch.zeros_like(to_scatter[0])  # will be filled with gradient summed across workers
        dist.reduce_scatter(output, to_scatter + tail, group=self.process_group)
        return output

    def assert_idle(self) -> None:
        """Assert we are in the idle state."""
        assert (
            self.training_state == TrainingState.IDLE
        ), f"wrong state to call no_sync. current state is {self.training_state}"


@torch.no_grad()
def cast_inputs_to_fp16(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """
    Cast any Tensors in *args or **kwargs to FP16.

    Doesn't currently support Tensors nested inside containers (e.g., dict).
    """
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    tensor_inputs, packed_non_tensor_inputs = split_non_tensors(flat_args)
    tensor_inputs = tuple(t.half() if torch.is_floating_point(t) else t for t in tensor_inputs)
    flat_args = unpack_non_tensors(tensor_inputs, packed_non_tensor_inputs)
    args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
    return args, kwargs


def cast_buffers_(
    module: nn.Module, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> None:
    """Cast all of module.named_buffers to device, dtype."""
    # if buffers are already on the right device and/or dtype this is just python loop cost
    for key, buf in module.named_buffers(recurse=False):
        if buf is not None:
            setattr(module, key, buf.to(dtype=dtype, device=device))


def free_storage_(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        assert data.storage().size() == data.numel()
        data.storage().resize_(0)


@torch.no_grad()
def alloc_storage_(data: torch.Tensor, size: torch.Size) -> None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():  # no need to reallocate
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())
