# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple

import torch
from torch import nn


def _conditional_amp_fwd_decorator(orig_func):  # type: ignore

    if hasattr(torch.cuda.amp, "custom_fwd"):
        return torch.cuda.amp.custom_fwd(orig_func)  # type: ignore

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) -> Any:
        return orig_func(*args, **kwargs)

    return inner_decorator


def _conditional_amp_bwd_decorator(orig_func):  # type: ignore
    if hasattr(torch.cuda.amp, "custom_bwd"):
        return torch.cuda.amp.custom_bwd(orig_func)  # type: ignore

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) -> Any:
        return orig_func(*args, **kwargs)

    return inner_decorator


def _split(modules: nn.Sequential, number_splits: int) -> List[List[nn.Module]]:
    number_splits = min(len(modules), number_splits)
    splits: List[List[nn.Module]] = [[] for _ in range(number_splits)]

    # Count the number of parameters per exposed layer, use that as a proxy for memory footprint
    total_number_params = sum([sum(p.numel() for p in m.parameters()) for m in modules])
    number_parameters_per_shard = total_number_params // number_splits

    current_shard = 0

    logging.info(
        f"This model has {total_number_params/1e6:.2f}M parameters, aiming for {number_parameters_per_shard/1e6:.2f}M parameters per shard"
    )

    for m in modules:
        for p in m.parameters():
            p.data = p.data.pin_memory()
        # Number of parameters in the current shard
        current_shard_params = sum(p.numel() for sm in splits[current_shard] for p in sm.parameters())

        # This shard is big enough, point to the next one
        if (
            current_shard_params > 0
            and current_shard_params + sum(p.numel() for p in m.parameters()) > number_parameters_per_shard
            and current_shard < number_splits - 1
        ):
            current_shard += 1

        splits[current_shard].append(m)

    for i, split in enumerate(splits):
        current_shard_params = sum(p.numel() for sm in split for p in sm.parameters())
        logging.info(f"Shard {i} holds {current_shard_params/1e6:.2f}M parameters")

    return splits


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the 
    fly for the FW and BW pass on the given device.
    """

    def __init__(
        self, cpu_model_shard: nn.Module, device: torch.device, offload_device: torch.device, index: int,
    ):
        super().__init__()
        self.model_shard = cpu_model_shard
        self.index = index

        # Save all the parameter sizes to be able to restore them
        self.device = device
        torch.cuda.device(self.device)

        self.offload_device = offload_device

        self.model_shard.to(offload_device)
        self._cpu_to_gpu_stream = torch.cuda.Stream(device=self.device)
        self._gpu_to_cpu_stream = torch.cuda.Stream(device=self.device)

    def forward(self, *inputs):  # type: ignore
        return self.model_shard(*inputs) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def to(self, device: torch.device) -> "ModelShard":  # type: ignore
        # Make sure that the lookahead and lookback shards are not captured by this call
        self.model_shard.to(device)
        return self

    def train(self, mode: bool = True) -> "ModelShard":
        # Make sure that the lookahead and lookback shards are not captured by this call
        self.model_shard.train(mode)
        return self

    def to_device(self) -> None:
        self.model_shard.to(device=self.device, non_blocking=True)

    def forward_load(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            # Restore all the parameter buffers
            self.model_shard.to(device=self.device, non_blocking=non_blocking)

    def backward_load(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard.to(self.device, non_blocking=non_blocking)

    def forward_drop(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)

    def backward_drop(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)


class ActivationCheckpointing(torch.autograd.Function):
    """
     This Function enables checkpointing of intermediate activations at
     shard boundaries by overriding the forward and backward pass of the nn.Module.

     - In the FW pass, it drops parameters in the previous shard and
     loads parameters for the next shard. No graph is constructed in the FW pass.
     This enables us to offload intermediate activations present at the shard
     boundaries.

     - In the BW pass, it does the reverse. We run the forward pass using the 
     saved intermediate activations and calculate gradients as needed. 
     The trade-off is latency vs memory when using activation checkpointing.

     - Follows heavily from https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint.

     NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
     """

    @staticmethod
    @_conditional_amp_fwd_decorator  # type: ignore
    def forward(ctx: Any, inputs: Any, dummy_input: Any, model_instance: Any) -> Any:
        inputs = inputs if isinstance(inputs, tuple) else (inputs,)

        ctx.inputs = inputs
        ctx.model_instance = model_instance
        # TODO(anj-s): We might need to store this for each boundary activation.
        # Currently we assume all boundary activation inputs require
        ctx.grad_requirements = tuple(x.requires_grad for x in inputs)
        ctx.fwd_rng_state = torch.get_rng_state()

        # List of input activations starting with the given input.
        model_instance._activations = [inputs]
        # Enumerate through layer shards and apply activations from the previous shard.
        for index, layer_shard in enumerate(model_instance.model_slices):
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:forward_load"):
                # Bring in the current activations onto the device.
                model_instance._activations[index] = tuple([a.cuda() for a in list(model_instance._activations[index])])
                # Bring in the current layer shard onto the device.
                layer_shard.forward_load()

            # Apply the FP and store the activations on the CPU.
            inputs = model_instance._activations[index]
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:no_grad_forward_pass"):
                with torch.no_grad():
                    output_list: List[Any] = []
                    for given_input in inputs:
                        given_input_list = torch.chunk(given_input, model_instance._num_microbatches)
                        given_output_list = []
                        for inputs in given_input_list:
                            output = layer_shard(inputs)
                            given_output_list.append(output)
                        given_output = torch.cat(given_output_list).squeeze(-1)
                        output_list.append(given_output)
                    output = tuple(output_list)

            output = output if isinstance(output, tuple) else (output,)
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:forward_drop"):
                # The last instance will lose the gradient function if we move it to the CPU.
                # This is because all grad function are present on the device that ran the FW pass.
                if index == len(model_instance.model_slices) - 1:
                    model_instance._activations.append(output)
                else:
                    model_instance._activations.append(tuple([a.cpu() for a in list(output)]))
                # Move the layer shard back to the CPU.
                layer_shard.forward_drop()

        # TODO(anj-s): Check device of the result to make sure the outputs and targets match device.
        result = model_instance._activations[-1]
        result = [r.cuda() for r in result]
        for r in result:
            r.requires_grad = True
        return result[0] if len(result) == 1 else result

    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):  # type: ignore
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.inputs
        model_instance = ctx.model_instance

        for i, need_grad in enumerate(ctx.grad_requirements):
            inputs[i].requires_grad = need_grad

        all_grads = [grad_outputs]

        for model_shard, activation in zip(
            reversed(model_instance.model_slices), reversed(model_instance._activations[:-1])
        ):
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:backward_load"):
                # Move the model shard to the device.
                model_shard.backward_load()

            # Store the BW pass state.
            bwd_rng_state = torch.get_rng_state()

            # TODO(anj-s): Why detach inputs?
            activation = torch.utils.checkpoint.detach_variable(activation)
            # Get the last gradient calculation.
            final_grads = all_grads[-1]

            if isinstance(activation, torch.Tensor):
                activation = (activation,)
            if isinstance(final_grads, torch.Tensor):
                final_grads = (final_grads,)
            # Iterate through all the inputs/outputs of a shard (there could be multiple).
            chunked_grad_list: List[Any] = []
            # Chunk the activation and grad based on the number of microbatches that are set.
            for chunked_activation, chunked_grad in zip(
                torch.chunk(*activation, model_instance._num_microbatches),  # type: ignore
                torch.chunk(*final_grads, model_instance._num_microbatches),  # type: ignore
            ):
                # Set the states to what it used to be before the forward pass.
                torch.set_rng_state(ctx.fwd_rng_state)

                if isinstance(chunked_activation, torch.Tensor):
                    chunked_activation = (chunked_activation,)  # type: ignore
                if isinstance(chunked_grad, torch.Tensor):
                    chunked_grad = (chunked_grad,)  # type: ignore

                # Since we need a grad value of a non leaf element we need to set these properties.
                for a in chunked_activation:
                    if a.dtype == torch.long:
                        continue
                    a.requires_grad = True
                    a.retain_grad()

                with torch.autograd.profiler.record_function(
                    "fairscale.experimental.nn.offload:forward_pass_with_enable_grad"
                ):
                    with torch.enable_grad():
                        # calculate the output of the last shard wrt to the stored activation at the slice boundary.
                        outputs = model_shard(*chunked_activation)

                # Set the states back to what it was at the start of this function.
                torch.set_rng_state(bwd_rng_state)
                with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:backward_pass"):
                    torch.autograd.backward(outputs, chunked_grad)
                intermediate_grads = []
                for a in chunked_activation:
                    if a.grad is not None:
                        intermediate_grads.append(a.grad)
                if None not in intermediate_grads:
                    chunked_grad_list += intermediate_grads
            if chunked_grad_list:
                # Append the list of grads to the all_grads list and this should be on the CPU.
                all_grads.append(torch.cat(chunked_grad_list).squeeze(-1))  # type: ignore
            # TODO(anj-s): Why does moving activations to CPU cause the .grad property to be None?
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:backward_drop"):
                # Move the shard back to the CPU.
                model_shard.backward_drop()
        detached_inputs = model_instance._activations[0]
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


class OffloadModel(nn.Module):
    """Wrapper used offload parts of a model to the CPU.

    The model is sharded into chunks and at each iteration, a
    single chunk is copied from CPU->GPU, FW pass is computed and
    the chunk is copied back to CPU. This process is repeated for
    all the chunks. In the BW pass, the same process happens in
    reverse.

    Note: OffloadModel currently only supports nn.Sequential models.

    Args:
        module (~torch.nn.Sequential): Module to be offloaded.

        device (torch.device):
            Device where the active model should reside.

        offload_device (torch.device):
            Device where the inactive model should reside.

        num_slices (int):
            Number of slices into which the model should be chunked.

        checkpoint_activation (bool):
            Boolean to indicate if we want to checkpoint intermediate
            activation states on the CPU. Default value is False.

        num_microbatches (int):
            Number of microbatches which should be run per model
            shard on device.
    """

    def __init__(
        self,
        model: nn.Sequential,
        device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
        num_slices: int = 3,
        checkpoint_activation: bool = False,
        num_microbatches: int = 1,
    ):
        super().__init__()
        if not model:
            raise TypeError("`model` argument to `OffloadModel` cannot be None.")

        if not device:
            raise TypeError("`device` argument to `OffloadModel` cannot be None.")

        if not isinstance(model, nn.Sequential):
            raise TypeError("`model` argument to `OffloadModel` must be of type `nn.Sequential`.")

        if not torch.cuda.is_available():
            raise TypeError("CUDA must be available as one of the compute devices for `OffloadModel`.")

        self.device = device
        self.offload_device = offload_device

        # Slice the model into roughly equivalent sequential shards.
        splits = _split(model, num_slices)

        # List of model shards that will be placed on/off the device.
        self.model_slices: List[nn.Module] = []

        for i, split in enumerate(splits):
            # Add one model handling this slice
            self.model_slices.append(
                ModelShard(
                    cpu_model_shard=nn.Sequential(*split), device=device, offload_device=offload_device, index=i,
                )
            )

        # Expose a unified view of the slices
        self._model = torch.nn.Sequential(*self.model_slices)

        # intermediate activations at the slice boundaries.
        self._activations: List[Tuple] = []

        # Currently we only support microbatches with activation checkpointing.
        if not checkpoint_activation and num_microbatches > 1:
            raise RuntimeError("We currently only support microbatches with activation checkpointing.")

        # Bool indicating if we want to checkpoint activation on the host.
        self._checkpoint_activation = checkpoint_activation

        # Number of microbatches to run per batch on the device
        self._num_microbatches = num_microbatches

    def forward(self, *inputs: Any, **_: Any) -> Any:
        # `apply` calls the `forward` function of the `ActivationCheckpointing` class
        # and the `forward` function calls `inputs` on the first model shard.
        # Please see https://pytorch.org/docs/stable/autograd.html#function for more details.

        # We need the second param to be a dummy input to enable the
        # backward pass to be triggered for integer inputs.
        return ActivationCheckpointing.apply(*inputs, torch.tensor([], requires_grad=True), self)
