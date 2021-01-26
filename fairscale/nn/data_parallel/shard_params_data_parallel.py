# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from fairscale.nn.misc import FlattenParamsWrapper
from fairscale.utils.containers import (
    apply_to_tensors,
    pack_kwargs,
    split_non_tensors,
    unpack_kwargs,
    unpack_non_tensors,
)


class ShardParamsDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters.

    Usage::

        sharded_module = ShardParamsDistributedWrapper(my_module)
        x = sharded_module(x, y=3, z=torch.Tensor([1]))
        loss = x.sum()
        loss.backward()

    It is also possible to shard individual layers separately and have an outer
    wrapper handle any leftover parameters::

        model = nn.Sequential(
            nn.Linear(5, 100),
            ShardParamsDistributedWrapper(nn.Linear(100, 100)),
            ShardParamsDistributedWrapper(nn.Linear(100, 100)),
            nn.Linear(100, 5),
        )
        sharded_model = ShardParamsDistributedWrapper(model)
        x = sharded_model(x)
        loss = x.sum()
        loss.backward()

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
        process_group=None,
        reshard_after_forward: bool = True,
        mixed_precision: bool = False,
        fp32_reduce_scatter: bool = False,
        flatten_parameters: bool = True,
        cpu_offload: bool = False,
        compute_device: Optional[torch.device] = None,
        compute_dtype: Optional[torch.dtype] = None,
        move_grads_to_cpu: Optional[bool] = False,
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
        self.move_grads_to_cpu = move_grads_to_cpu

        if self.fp32_reduce_scatter and not self.mixed_precision:
            raise ValueError("fp32_reduce_scatter requires mixed_precision=True")
        if self.cpu_offload and not self.mixed_precision:
            raise ValueError("cpu_offload requires mixed_precision=True")

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        params = list(p for p in module.parameters() if not getattr(p, "_is_sharded", False))

        if self.flatten_parameters and len(params) > 0:
            self.module = FlattenParamsWrapper(module, param_list=params)
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

    @torch.no_grad()
    def _shard_initial_params(self):
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
            p.data = p.data.view(-1)[s:e].clone()
            free_storage_(orig_data)

    def __getattr__(self, name):
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        state["orig_sizes"] = [p._orig_size for p in self.params]
        if state["process_group"] is not None:
            state["process_group"] = "MISSING"  # raise error if used
        if "_fp32_to_fp16_stream" in state:
            del state["_fp32_to_fp16_stream"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        def fixup(p, size):
            assert isinstance(p, torch.nn.Parameter)
            p.data = p.data.clone()  # move tensors out of shared memory
            p._is_sharded = True
            p._orig_size = size
            return p

        self.params = [fixup(p, size) for p, size in zip(self.params, self.orig_sizes)]
        del self.orig_sizes

    def state_dict(self, *args, **kwargs):
        """
        Returns the whole (unsharded) state of the module. Parameters are not
        sharded, so the resulting state_dict can be loaded directly by the
        wrapped Module without any sharding-specific logic.
        """
        torch.cuda.synchronize()
        self._rebuild_full_params()
        # We don't free the params after generating the state dict, since
        # freeing is done in-place (via the Storagee) and would corrupt the
        # returned state dict.
        return self.module.state_dict(*args, **kwargs)

    def local_state_dict(self, *args, **kwargs):
        """
        Returns the local (sharded) state of the module. Parameters are sharded,
        so the resulting state_dict can only be loaded after the Module has been
        wrapped with ShardParamsDistributedWrapper.
        """
        if self.flatten_parameters:
            kwargs["unflatten_params"] = False
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Load a whole (unsharded) state_dict."""
        self._rebuild_full_params()
        output = self.module.load_state_dict(*args, **kwargs)
        self._free_full_params()
        return output

    def load_local_state_dict(self, *args, **kwargs):
        """Load a local (sharded) state_dict."""
        return self.module.load_state_dict(*args, **kwargs)

    @torch.no_grad()
    def _pre_forward_init(self):
        did_init = False
        for p in self.params:
            if not hasattr(p, "_full_param"):
                did_init = True
                assert p._is_sharded

                p._fp32_shard = p.data

                if self.mixed_precision:
                    assert p._fp32_shard.dtype == torch.float32
                    if self.cpu_offload:
                        p._fp32_shard = p._fp32_shard.pin_memory()
                    p._fp16_shard = torch.zeros_like(
                        p._fp32_shard, device=self.compute_device, dtype=self.compute_dtype,
                    )
                    free_storage_(p._fp16_shard)
                    p._full_param = torch.zeros(p._orig_size, device=self.compute_device, dtype=self.compute_dtype)
                else:
                    p._fp16_shard = None  # use _fp32_shard
                    p._full_param = p._fp32_shard.new_empty(p._orig_size)

                p._full_param = p._full_param.to(dtype=self.compute_dtype, device=self.compute_device)
                free_storage_(p._full_param)

                p.data = p._fp32_shard

                if self.move_grads_to_cpu:
                    if self.mixed_precision and not self.fp32_reduce_scatter:
                        grad_dtype = torch.float16
                    else:
                        grad_dtype = torch.float32
                    p._cpu_grad = torch.zeros_like(p.data, dtype=grad_dtype, device="cpu").pin_memory()

        if did_init:
            self._fp32_to_fp16_stream = torch.cuda.Stream()
        self._fp32_to_fp16_stream.wait_stream(torch.cuda.current_stream())

    def forward(self, *args, **kwargs):
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

        # Register pre-backward hook to run before the wrapped module's backward.
        if torch.is_grad_enabled():
            pre_backward_hook_has_run = [False]

            def _pre_backward_hook(*unused):
                if pre_backward_hook_has_run[0]:
                    return  # only run once
                pre_backward_hook_has_run[0] = True

                if self.reshard_after_forward:
                    self._rebuild_full_params()
                else:
                    self._use_full_params()

            def _register_hook(t):
                t.register_hook(_pre_backward_hook)
                return t

            # Attach hooks to Tensor outputs.
            outputs = apply_to_tensors(_register_hook, outputs)

        return outputs

    def _register_post_backward_hooks(self):
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
    def _post_backward_hook(self, param, *unused):
        if param.grad is None:
            return
        if param.grad.requires_grad:
            raise RuntimeError("ShardParamsDistributedWrapper only works with gradients that don't require grad")

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
        param.grad.data = self._reduce_scatter(param.grad.data.view(-1))

        if self.move_grads_to_cpu:
            param._cpu_grad.copy_(param.grad.data, non_blocking=True)
            param.grad.data = param._cpu_grad

        # Cast grad to param's dtype (typically FP32).
        if self.mixed_precision:
            param.grad.data = param.grad.data.to(dtype=param.data.dtype)

    @torch.no_grad()
    def _rebuild_full_params(self):
        if self.mixed_precision:
            self._cast_fp32_param_shards_to_fp16()

        for p in self.params:
            # All-gather parameters
            alloc_storage_(p._full_param, size=p._orig_size)
            output_list = list(p._full_param.view(-1).chunk(self.world_size))
            dist.all_gather(output_list, p.data, group=self.process_group)
            p.data = p._full_param
            p.grad = None

            if self.mixed_precision:
                self._free_fp16_param_shard([p])

    @torch.no_grad()
    def _use_full_params(self):
        for p in self.params:
            assert p._full_param.storage().size() != 0
            p.data = p._full_param

    @torch.no_grad()
    def _free_full_params(self, params=None):
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
    def _use_fp32_param_shard(self, params=None):
        if params is None:
            params = self.params
        for p in params:
            p.data = p._fp32_shard

    @torch.no_grad()
    def _cast_fp32_param_shards_to_fp16(self, params=None):
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
    def _free_fp16_param_shard(self, params=None):
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
    def _reduce_scatter(self, tensor, output=None):
        assert tensor.numel() % self.world_size == 0
        tensor = tensor.view(self.world_size, -1)
        if output is None:
            output = torch.zeros_like(tensor[0])
        dist.reduce_scatter(output, list(tensor.unbind(0)), group=self.process_group)
        return output


@torch.no_grad()
def cast_inputs_to_fp16(*args, **kwargs):
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


def cast_buffers_to_fp16(module):
    for key, buf in module.named_buffers(recurse=False):
        if buf is not None:
            setattr(module, key, buf.half())


def free_storage_(data):
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        assert data.storage().size() == data.numel()
        data.storage().resize_(0)


@torch.no_grad()
def alloc_storage_(data, size):
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())
    # data.set_(size=size)
