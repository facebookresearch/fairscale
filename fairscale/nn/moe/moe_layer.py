# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        world_size = dist.get_world_size(group)
        input = input.contiguous()
        output = torch.empty_like(input)
        input_chunks = list(input.chunk(world_size))
        output_chunks = list(output.chunk(world_size))
        dist.all_to_all(output_chunks, input_chunks, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


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

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True  # type: ignore

    def all_to_all_dispatch(self, dispatch_mask: Tensor, input: Tensor) -> Tensor:
        dispatched_input = torch.einsum("gsec,gsm->egcm", dispatch_mask.float(), input)
        return _AllToAll.apply(self.group, dispatched_input)

    def all_to_all_combine(self, combine_weights: Tensor, input: Tensor) -> Tensor:
        expert_output = _AllToAll.apply(self.group, input)
        return torch.einsum("gsec,egcm->gsm", combine_weights, expert_output)

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 4, "input Tensor must have dimensions: (g)roup, (s)equence, (t)oken, (m)odel"
        assert input[0].shape[0] == len(self.experts), "group dimension size must be equal to number of local experts"

        # Implement Algorithm 2 from GShard paper.
        shape = input[0].shape
        # Reshape into S tokens per group.
        reshaped_input = input[0].reshape(shape[0], -1, shape[3])
        self.l_aux, combine_weights, dispatching_mask = self.gate(reshaped_input)
        dispatched_input = self.all_to_all_dispatch(dispatching_mask, reshaped_input)
        assert dispatched_input.shape[1] == len(self.experts)
        chunks = dispatched_input.chunk(len(self.experts), dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        combined_output = self.all_to_all_combine(combine_weights, expert_output)
        return combined_output.reshape(shape)
