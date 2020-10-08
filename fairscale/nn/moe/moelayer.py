# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn import Module


class MOELayer(Module):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        l_aux, combine_weights, dispatch_mask = moe(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    gate: Module
    expert: Module
    l_aux: Tensor

    def __init__(self, gate: Module, expert: Module) -> None:
        super().__init__()
        self.gate = gate
        self.expert = expert

    def all_to_all_dispatch(self, dispatch_mask: Tensor, input: Tensor) -> Tensor:
        dispatched_input = torch.einsum("gsec,gsm->egcm", dispatch_mask.float(), input)
        # TODO(msb) all-to-all
        dispatched_input = torch.squeeze(dispatched_input, 0)  # drop E dimension
        return dispatched_input

    def all_to_all_combine(self, combine_weights: Tensor, input: Tensor) -> Tensor:
        # TODO(msb) all-to-all
        expert_output = torch.unsqueeze(input, 1)  # add E dimension
        output = torch.einsum("gsec,gecm->gsm", combine_weights, expert_output)
        return output

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        # Implement Algorithm 2 from GShard paper.
        shape = input.shape
        # Reshape into S tokens per group.
        reshaped_input = input.reshape(shape[0], -1, shape[3])
        self.l_aux, combine_weights, dispatching_mask = self.gate(reshaped_input)
        dispatched_input = self.all_to_all_dispatch(dispatching_mask, reshaped_input)
        expert_output = self.expert(dispatched_input)
        combined_output = self.all_to_all_combine(combine_weights, expert_output)
        return combined_output.reshape(shape)
