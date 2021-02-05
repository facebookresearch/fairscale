# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A wrapper which streams the model in and out of the GPU automatically during FW and optionally BW passes
(Can be used for inference only)
"""

from builtins import isinstance
import logging
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn


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
    Wrap one shard of the model, make it possible to load parameters on the fly for the FW pass and gather gradients.
    Depending on whether this rank is or is not the `owner_rank`, this ModelShard either only handles
    a shard of the compute and is stateless or also owns the up to date state.
    """

    def __init__(
        self, cpu_model_shard: nn.Module, device: torch.device, offload_device: torch.device,
    ):
        super().__init__()
        self.model_shard = cpu_model_shard

        # Save all the parameter sizes to be able to restore them
        self.device = device
        torch.cuda.device(self.device)

        self.offload_device = offload_device

        self.model_shard.to(offload_device)
        self.cuda_stream = torch.cuda.Stream(
            device=self.device
        )  # needed to make sure load/offload really run in parallel with compute

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
        with torch.cuda.stream(self.cuda_stream):
            # Restore all the parameter buffers
            self.model_shard.to(device=self.device, non_blocking=non_blocking)

    def backward_load(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self.cuda_stream):
            self.model_shard.to(self.device, non_blocking=non_blocking)

    def forward_drop(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self.cuda_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)

    def backward_drop(self, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self.cuda_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)


class ShardSyncLayer(torch.autograd.Function):
    """
     The shard sync layer is a synchronization point between model shards.

     - In the forward pass, it drops parameters in the previous shard and
     loads parameters for the next shard.

     - In the backward pass, it does the reverse.

     It does not change or create any outputs at all, instead it just
     forwards the input as the output.

     NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
     """

    @staticmethod
    def forward(ctx: Any, p2_shard: Optional[ModelShard], p1_shard: Optional[ModelShard], n1_shard: Optional[ModelShard], n2_shard: Optional[ModelShard], *inputs: Any) -> Any:  # type: ignore
        # Drop the shard we just went through, except if this is the last one in line
        if p1_shard and n1_shard:
            p1_shard.forward_drop(non_blocking=True)

        # Start the load of the next shard in line, opportunistically look ahead
        if n2_shard:
            n2_shard.forward_load(non_blocking=True)

        ctx.p2_shard = p2_shard
        ctx.p1_shard = p1_shard
        ctx.n1_shard = n1_shard
        ctx.n2_shard = n2_shard

        # FIXME: handle corner cases / type dependent
        outputs = inputs

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        if ctx.n1_shard:
            ctx.n1_shard.backward_drop(non_blocking=True)

        # Opportunistically pre-load ahead of the compute wavefront
        if ctx.p2_shard:
            ctx.p2_shard.backward_load(non_blocking=True)

        # The returned variables need to mirror the forward inputs
        if isinstance(grad_outputs, tuple):
            return None, None, None, None, grad_outputs[0]

        return None, None, None, None, grad_outputs


class OffloadWrapperExperimental(nn.Module):
    """Implements training with optimizer state sharding and model sharding.

    This experiments with a different way to get to the full zero suite
    The model is sharded, then the normal distributed data parallel algorithm can be used on a per-model shard basis.
    Each shard is offloaded and loaded following a compute wavefront, during the forward and backward pass.

    All the gradients are centralized on a given rank (which is model-shard dependent, so that the gradients
    redundancy can be removed). Each model shard can be updated by a normal pytorch optimizer.

    Args:
        module (~torch.nn.Sequential): module to be parallelized
        optimizer (~torch.optim.Optimizer): optimizer to be used for training
        optimizer_params(Dict): extra parameters for the optimizer

        device (torch.device):
            device where the active model should reside

        offload_device (torch.device):
            device where the inactive model should reside

        n_slices (int):
            number of slices in which to decomppose the model
    """

    def __init__(
        self,
        model_cpu: nn.Sequential,  # hard pre-requisite for now, easier model slicing
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict[str, Any],
        device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
        n_slices: int = 5,
    ):
        super().__init__()

        self.device = device
        self.offload_device = offload_device

        # Slice the model into roughly equivalent sequential shards
        splits = _split(model_cpu, n_slices)

        # Each rank either owns the slice, or temporarily helps processing it in a data parallel fashion
        self.model_slices: List[nn.Module] = []

        for split in splits:
            # Add one model handling this slice
            self.model_slices.append(
                ModelShard(cpu_model_shard=nn.Sequential(*split), device=device, offload_device=offload_device,)
            )

            # Use one normal optimizer per slice
            # TODO: Keep all optimizers, return a wrap which will distribute the steps()
            self.optimizer = optimizer(nn.Sequential(*split).parameters(), **optimizer_params)

        # Expose a unified view of the slices
        self.model = torch.nn.Sequential(*self.model_slices)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        # Slice per slice FW, sync in between
        syncRanks = ShardSyncLayer.apply

        # TODO: Rewrite this and make it more flexible, this is ugly
        for i, (p2, p1, n1, n2) in enumerate(
            zip(
                [None, None, *self.model_slices],
                [None, *self.model_slices],
                [*self.model_slices, None],
                [*self.model_slices, None, None],
            )
        ):
            print(i)

            # Per shard FW
            inputs = p1(*inputs) if p1 else inputs

            # Call the custom autograd hooks (discard/load slices FW and BW)
            inputs = syncRanks(p2, p1, n1, n2, *inputs)

        return inputs[0] if len(inputs) == 1 else inputs
