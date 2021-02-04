# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.nn as nn
from torch.nn import Parameter

from fairscale.nn.misc import FlattenParamsWrapper
from fairscale.utils.containers import (
    apply_to_tensors,
    pack_kwargs,
    split_non_tensors,
    unpack_kwargs,
    unpack_non_tensors,
)

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class ShardParamsDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters.

    Usage::

        sharded_module = ShardParamsDataParallel(my_module)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        x = sharded_module(x, y=3, z=torch.Tensor([1]))
        loss = x.sum()
        loss.backward()
        optim.step()

    It is also possible to shard individual layers separately and have an outer
    wrapper handle any leftover parameters. This can be helpful to further
    reduce memory usage and to improve training speed by distributing the
    unsharding (all-gather) across the forward pass. For example::

        sharded_model = ShardParamsDataParallel(
            nn.Sequential(
                nn.Linear(5, 100),
                ShardParamsDataParallel(nn.Linear(100, 100)),
                ShardParamsDataParallel(nn.Linear(100, 100)),
                nn.Linear(100, 5),
            )
        )

    Args:
        module (nn.Module): module to checkpoint
        process_group (Optional): process group for sharding
        reshard_after_forward (bool, Optional): if True, reshard parameters
            after the forward pass. This saves memory but slows training.
        mixed_precision (bool, Optional): if True, inputs, activations and
            gradients will be kept in FP16; computation and communication will
            occur in FP16; and a (sharded) master copy of the model weights will
            be maintained in FP32.
        fp32_reduce_scatter (bool, Optional): if True, then reduce-scatter
            gradients in FP32. This is only relevant when *mixed_precision* is
            ``True``.
        flatten_parameters (bool, Optional): if True, flatten parameters into a
            single contiguous tensor, which improves training speed.
        cpu_offload (bool, Optional): if True, offload FP32 params to CPU. This
            is only relevant when *mixed_precision* is ``True``.
        compute_device (torch.device, Optional): device to move params to for
            computation. This is primarily relevant with *cpu_offload* and
            defaults to "cuda".
        compute_dtype (torch.dtype, Optional): dtype for full parameters for
            computation. This defaults to torch.float32 unless *mixed_precision*
            is set, in which case it defaults to torch.float16.
        move_grads_to_cpu (bool, Optional): move grad shard to CPU after
            reduction. This is useful when combined with CPU-based optimizers.
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
        compute_device: Optional[torch.device] = None,
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
        self.compute_device = compute_device or torch.device("cuda")
        self.compute_dtype = compute_dtype or (torch.float16 if mixed_precision else torch.float32)
        self.move_grads_to_cpu = cpu_offload if move_grads_to_cpu is None else move_grads_to_cpu

        if self.fp32_reduce_scatter and not self.mixed_precision:
            raise ValueError("fp32_reduce_scatter requires mixed_precision=True")
        if self.cpu_offload and not self.mixed_precision:
            raise ValueError("cpu_offload requires mixed_precision=True")

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        params = list(p for p in module.parameters() if not getattr(p, "_is_sharded", False))

        if self.flatten_parameters and len(params) > 0:
            self.module: nn.Module = FlattenParamsWrapper(module, param_list=params)
            del module
            self.params = [self.module.flat_param]
        else:
            self.module = module
            self.params = params

        # Shard module parameters.
        self._shard_initial_params()

        if self.mixed_precision:
            # Cast all module buffers to FP16 (buffers are not sharded).
            self.apply(cast_buffers_to_fp16)

        # Make sure all parameters are sharded.
        for n, p in self.named_parameters():
            assert getattr(p, "_is_sharded", False), f"found unsharded parameter: {n} ; {p.size()}"

        # Flag to indicate if this instance is wrapped by any other
        # ShardParamsDataParallel instances. This flag is only set after the
        # first forward pass.
        self._is_root: Optional[bool] = None

    @torch.no_grad()
    def _shard_initial_params(self) -> None:
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

            shard_size = p.data.numel() // self.world_size
            s = self.rank * shard_size
            e = (self.rank + 1) * shard_size

            orig_data = p.data
            p.data = torch.flatten(p.data)[s:e].clone()
            free_storage_(orig_data)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def __getstate__(self) -> Dict[str, str]:
        state = copy.copy(self.__dict__)
        state["orig_sizes"] = [p._orig_size for p in self.params]
        if state["process_group"] is not None:
            state["process_group"] = "MISSING"  # process_group isn't pickleable
        if "_fp32_to_fp16_stream" in state:
            del state["_fp32_to_fp16_stream"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Intercept state setting and perform needed changes on params."""
        super().__setstate__(state)

        def fixup(p: Parameter, size: torch.Size) -> Parameter:
            assert isinstance(p, Parameter)
            p.data = p.data.clone()  # move tensors out of shared memory
            # Ignore mypy error since we add additional fields to a param.
            p._is_sharded = True
            p._orig_size = size
            return p

        self.params = [fixup(p, size) for p, size in zip(self.params, self.orig_sizes)]
        del self.orig_sizes

    # TODO (Min): figuring out how to do typing for this overloaded function.
    def state_dict(self, *args, **kwargs):  # type: ignore
        """
        Returns the whole (unsharded) state of the module. Parameters are not
        sharded, so the resulting state_dict can be loaded directly by the
        wrapped Module without any sharding-specific logic.
        """
        torch.cuda.synchronize()
        self._rebuild_full_params()
        # We don't free the params after generating the state dict, since
        # freeing is done in-place (via the Storage) and would corrupt the
        # returned state dict.
        return self.module.state_dict(*args, **kwargs)

    # TODO (Min): figuring out how to do typing for this overloaded function.
    def local_state_dict(self, *args, **kwargs):  # type: ignore
        """
        Returns the local (sharded) state of the module. Parameters are sharded,
        so the resulting state_dict can only be loaded after the Module has been
        wrapped with ShardParamsDataParallel.
        """
        if self.flatten_parameters:
            kwargs["unflatten_params"] = False
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """Load a whole (unsharded) state_dict."""
        self._rebuild_full_params()
        output = self.module.load_state_dict(state_dict, strict)
        self._free_full_params()
        return output

    def load_local_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """Load a local (sharded) state_dict."""
        return self.module.load_state_dict(state_dict, strict)

    @torch.no_grad()
    def _init_param(self, p: Parameter) -> None:
        assert p._is_sharded
        assert not hasattr(p, "_full_param")

        p._fp32_shard = p.data

        if self.mixed_precision:
            assert p._fp32_shard.dtype == torch.float32

            if self.cpu_offload:
                assert p._fp32_shard.device == torch.device("cpu")
                p._fp32_shard = p._fp32_shard.pin_memory()

            p._fp16_shard = torch.zeros_like(p._fp32_shard, device=self.compute_device, dtype=self.compute_dtype)
            free_storage_(p._fp16_shard)
        else:
            p._fp16_shard = None  # use _fp32_shard

        p.data = p._fp32_shard

        p._full_param = torch.zeros(p._orig_size, device=self.compute_device, dtype=self.compute_dtype)
        free_storage_(p._full_param)

        if self.move_grads_to_cpu:
            if self.mixed_precision and not self.fp32_reduce_scatter:
                grad_dtype = torch.float16
            else:
                grad_dtype = torch.float32
            p._cpu_grad = torch.zeros_like(p.data, dtype=grad_dtype, device="cpu").pin_memory()

    @torch.no_grad()
    def _pre_forward_init(self) -> None:
        first_time_params = [p for p in self.params if not hasattr(p, "_full_param")]
        for p in first_time_params:
            self._init_param(p)

        if len(first_time_params) > 0:
            if self._is_root is None:
                # This implies that no other ShardParamsDataParallel instance
                # wraps this instance, otherwise it would have already set this
                # flag to False.
                self._is_root = True

                # As the root, we now set all children instances to False.
                for n, m in self.named_modules():
                    if n != "" and isinstance(m, ShardParamsDataParallel):
                        assert m._is_root is None
                        m._is_root = False

            if self._is_root:
                # Stream for moving FP32 master params (which may be on CPU) to
                # FP16 for computation. We share this stream with all children
                # instances, which allows them to overlap transfers across the
                # forward pass without synchronizing with the default stream.
                self._fp32_to_fp16_stream = torch.cuda.Stream()

                for n, m in self.named_modules():
                    if n != "" and isinstance(m, ShardParamsDataParallel):
                        m._fp32_to_fp16_stream = self._fp32_to_fp16_stream

        assert self._is_root is not None
        if self._is_root:
            # The top-most (root) instance needs to synchronize with the default
            # stream, so we don't move the FP32 master weights prematurely.
            self._fp32_to_fp16_stream.wait_stream(torch.cuda.current_stream())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._pre_forward_init()

        if self.mixed_precision:
            args, kwargs = cast_inputs_to_fp16(*args, **kwargs)

        # All-gather full parameters.
        self._rebuild_full_params()

        # Register backward hooks to reshard params and reduce-scatter grads.
        # These need to be re-registered every forward pass.
        self._register_post_backward_hooks()

        outputs = self.module(*args, **kwargs)

        if self.reshard_after_forward:
            self._free_full_params()

        # Switch to FP32 param shard after forward so that the optimizer will be
        # initialized with the correct dtype and size.
        self._use_fp32_param_shard()

        if torch.is_grad_enabled():
            outputs = self._register_pre_backward_hooks(outputs)

        return outputs

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:

        # Register pre-backward hook to run before the wrapped module's backward.
        pre_backward_hook_has_run = [False]

        def _pre_backward_hook(*unused: Any) -> None:
            if pre_backward_hook_has_run[0]:
                return  # only run once
            pre_backward_hook_has_run[0] = True

            if self.reshard_after_forward:
                self._rebuild_full_params()
            else:
                self._use_full_params()

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            t.register_hook(_pre_backward_hook)
            return t

        # Attach hooks to Tensor outputs.
        outputs = apply_to_tensors(_register_hook, outputs)

        return outputs

    def _register_post_backward_hooks(self) -> None:
        # Register backward hooks to reshard params and reduce-scatter grads.
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
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
        if param.grad is None:
            return
        if param.grad.requires_grad:
            raise RuntimeError("ShardParamsDataParallel only works with gradients that don't require grad")

        # Free full params and switch to FP32 shard after backward.
        self._free_full_params([param])
        self._use_fp32_param_shard([param])

        if self.mixed_precision:
            self._free_fp16_param_shard([param])

            if self.fp32_reduce_scatter:
                # Cast grad to FP32.
                param.grad.data = param.grad.data.to(param.dtype)

        # Average grad by world_size for consistency with PyTorch DDP.
        param.grad.data.div_(self.world_size)

        # Reduce-scatter grad.
        param.grad.data = self._reduce_scatter(torch.flatten(param.grad.data))

        if self.move_grads_to_cpu:
            param._cpu_grad.copy_(param.grad.data, non_blocking=True)
            param.grad.data = param._cpu_grad

        # Cast grad to param's dtype (typically FP32).
        if self.mixed_precision:
            param.grad.data = param.grad.data.to(dtype=param.data.dtype)

    @torch.no_grad()
    def _rebuild_full_params(self) -> None:
        """Get all shards of params."""
        if self.mixed_precision:
            self._cast_fp32_param_shards_to_fp16()

        for p in self.params:
            # All-gather parameters
            alloc_storage_(p._full_param, size=p._orig_size)
            output_list = list(torch.flatten(p._full_param).chunk(self.world_size))
            dist.all_gather(output_list, p.data, group=self.process_group)
            p.data = p._full_param
            p.grad = None

            if self.mixed_precision:
                self._free_fp16_param_shard([p])

    @torch.no_grad()
    def _use_full_params(self) -> None:
        for p in self.params:
            assert p._full_param.storage().size() != 0
            p.data = p._full_param

    @torch.no_grad()
    def _free_full_params(self, params: Optional[List[Parameter]] = None) -> None:
        """Free up storage for full parameters."""
        if params is None:
            params = self.params
        current_stream = torch.cuda.current_stream()
        for p in params:
            # There may be external references to the Tensor Storage that we
            # can't modify, such as references that are created by
            # ctx.save_for_backward in the forward pass. Thus when we unshard
            # parameters, we should reuse the original Tensor Storage object
            # and unshard it in-place. For now, just resize the Storage to 0 to
            # save memory.
            p._full_param.record_stream(current_stream)
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
        with torch.cuda.stream(self._fp32_to_fp16_stream):
            for p in params:
                assert p._fp16_shard is not None
                alloc_storage_(p._fp16_shard, size=p._fp32_shard.size())
                p._fp16_shard.copy_(p._fp32_shard, non_blocking=True)
                p.data = p._fp16_shard
        torch.cuda.current_stream().wait_stream(self._fp32_to_fp16_stream)

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
    def _reduce_scatter(self, tensor: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert tensor.numel() % self.world_size == 0
        tensor = tensor.view(self.world_size, -1)
        if output is None:
            output = torch.zeros_like(tensor[0])
        dist.reduce_scatter(output, list(tensor.unbind(0)), group=self.process_group)
        return output


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


def cast_buffers_to_fp16(module: nn.Module) -> None:
    """Cast buffers of a module to FP16."""
    for key, buf in module.named_buffers(recurse=False):
        if buf is not None:
            setattr(module, key, buf.half())


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
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())
