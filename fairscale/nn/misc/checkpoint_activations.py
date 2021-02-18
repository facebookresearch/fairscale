# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from fairscale.utils.containers import pack_kwargs, split_non_tensors, unpack_kwargs, unpack_non_tensors


def checkpoint_wrapper(module: nn.Module, offload_to_cpu: bool = False) -> nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:

        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))

    Args:
        module (nn.Module):
            module to wrap
        offload_to_cpu (Optional, bool):
            whether to offload activations to CPU

    Returns:
        (nn.Module):
            wrapped module
    """
    module.forward = functools.partial(_checkpointed_forward, module.forward, offload_to_cpu)  # type: ignore
    return module


def _checkpointed_forward(original_forward: Any, offload_to_cpu: bool, *args: Any, **kwargs: Any) -> Any:
    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {"offload": offload_to_cpu}
    output = CheckpointFunction.apply(original_forward, parent_ctx_dict, kwarg_keys, *flat_args)
    if isinstance(output, torch.Tensor):
        return output
    else:
        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
        return output


def get_rng_state() -> Dict[str, Any]:
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.

    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling :func:`unpack_non_tensors`.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        run_function: Any,
        parent_ctx_dict: Dict[str, Any],
        kwarg_keys: Tuple[str, ...],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(x.cpu() for x in tensor_inputs)

        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs
            return outputs

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Tuple[Optional[Tensor], ...]:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")

        tensor_inputs: Tuple = ctx.saved_tensors
        tensor_inputs = checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = tuple(t.to(ctx.fwd_device[i]) for i, t in enumerate(tensor_inputs))
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = get_rng_state()

        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)
        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("None of the outputs have requires_grad=True, " "this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)
        return (None, None, None) + grads
