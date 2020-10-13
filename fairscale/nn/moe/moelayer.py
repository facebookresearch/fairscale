# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, expert: Module, group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        self.expert = expert
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)

    def all_to_all_dispatch(self, dispatch_mask: Tensor, input: Tensor) -> Tensor:
        dispatched_input = torch.einsum("gsec,gsm->egcm", dispatch_mask.float(), input)
        dispatched_input = dispatched_input.contiguous()
        chunks = list(dispatched_input.chunk(self.world_size))
        dist.all_to_all(chunks, chunks, self.group)
        return dispatched_input

    def all_to_all_combine(self, combine_weights: Tensor, input: Tensor) -> Tensor:
        expert_output = input.contiguous()
        chunks = list(expert_output.chunk(self.world_size))
        dist.all_to_all(chunks, chunks, self.group)
        output = torch.einsum("gsec,egcm->gsm", combine_weights, expert_output)
        return output

    def forward(self, *input: Any, **kwargs: Any) -> Tensor:
        # Implement Algorithm 2 from GShard paper.
        shape = input[0].shape
        # Reshape into S tokens per group.
        reshaped_input = input[0].reshape(shape[0], -1, shape[3])
        self.l_aux, combine_weights, dispatching_mask = self.gate(reshaped_input)
        dispatched_input = self.all_to_all_dispatch(dispatching_mask, reshaped_input)
        expert_output = self.expert(dispatched_input)
        combined_output = self.all_to_all_combine(combine_weights, expert_output)
        return combined_output.reshape(shape)
