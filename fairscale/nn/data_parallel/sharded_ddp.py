# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
A module wrapper to go with a Sharded Optimizer in order to handle targeted gradient reduction/gathering automatically.
"""

import logging
from typing import Any, Callable, List, Union

from torch import nn
import torch.distributed as dist
from torch.nn import Parameter

from fairscale.optim.oss import OSS


def _get_global_rank(group: Any, rank: int) -> int:
    return rank if group is dist.group.WORLD else dist.distributed_c10d._get_global_rank(group, rank)  # type: ignore


class ShardedDataParallel(nn.Module):
    """
    Wrap the model, and reduce the gradients to the right rank during the backward pass.

    - the partition is given by the sharded optimizer
    - wrap the base model with a model which knows where to reduce each gradient
    - add an autograd function which calls the model grad dispatch on the way back

     Args:
        base_model (nn.Module):
            model to be wrapped
        sharded_optimizer (OSS, or list of OSS):
            the sharded optimizer(s) which will decide the gradient partitioning
    Keyword Args:
        process_group (torch.nn.Optimizer):
            optimizer to shard (default: SGD)
        process_group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffers (bool):
            whether to broadcast model buffers in between ranks at the beginning of each forward pass
    """

    def __init__(
        self,
        base_model: nn.Module,
        sharded_optimizer: Union[OSS, List[OSS]],
        process_group: Any = None,
        broadcast_buffers: bool = True,
    ):
        super().__init__()

        self.base_model = base_model
        self.sharded_optimizers = [sharded_optimizer] if isinstance(sharded_optimizer, OSS) else sharded_optimizer
        self.broadcast_buffers = broadcast_buffers

        # Communication related attributes
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.reference_global_rank = _get_global_rank(self.process_group, 0)  # picking rank 0 as the reference
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = _get_global_rank(self.process_group, self.rank)

        # Expose the same attributes as PytorchDDP, some frameworks rely on them.
        # See https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel
        self.is_multi_device_module = len({p.device for p in self.base_model.parameters()}) > 1
        distinct_device_types = {p.device.type for p in self.base_model.parameters()}
        assert len(distinct_device_types) == 1, (
            "ShardedDataParallel's input module must be on "
            "the same type of devices, but input module parameters locate in {}."
        ).format(distinct_device_types)
        self.device_type = list(distinct_device_types)[0]

        # Look up where this parameter should be reduced to
        def find_rank(param: Parameter) -> int:
            for optim in self.sharded_optimizers:
                if param in optim.param_to_rank.keys():
                    return optim.param_to_rank[param]

            assert False, "This parameter is not present in an optimizer, this should not happen"

        logging.info(f"Rank {dist.get_rank(self.process_group)}: Registering hooks")

        # Fill in a look-up table per grad
        self._grad_to_rank = [
            _get_global_rank(self.process_group, find_rank(p))
            for p in filter(lambda x: x.requires_grad, self.base_model.parameters())
        ]

        self._grad_to_be_reduced = [True for _ in filter(lambda x: x.requires_grad, self.base_model.parameters())]
        self._grad_accs: List[Callable] = []
        self._setup_bw_hooks()

        # Make sure that all ranks start with the same model
        self.sync_all_params()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if self.broadcast_buffers:
            self.sync_buffers()

        # Reset all the grad reduce flags
        self._grad_to_be_reduced = [True for _ in self._grad_to_be_reduced]

        return self.base_model(*inputs)

    def reduce(self) -> None:
        logging.warning("This is not useful anymore, gradients have been reduced automatically with the backward pass")

    def sync_all_params(self) -> None:
        """
        Sync the complete model states in between the ranks
        """
        work_handles = [
            dist.broadcast(t, src=self.reference_global_rank, group=self.process_group, async_op=True)
            for t in self.base_model.state_dict().values()
        ]

        _ = list(map(lambda x: x.wait(), work_handles))

    def sync_buffers(self, non_blocking: bool = False) -> None:
        """
        Sync all the param buffers in between ranks.
        """
        for x in self.base_model.buffers(recurse=True):
            dist.broadcast(x.data, self.reference_global_rank, self.process_group, async_op=True)

    def _setup_bw_hooks(self) -> None:
        """
        Attach a reduce function to each grad-requiring parameter. This makes the gradient reduction automatic whenever there's a BW pass
        """

        parameters_with_grad = list(filter(lambda x: x.requires_grad, self.base_model.parameters()))

        # Build one hook per parameter
        def get_reduce_fn(index: int) -> Callable:
            def reduce(*unused: Any) -> None:
                param = parameters_with_grad[index]

                if param.grad is not None and self._grad_to_be_reduced[index]:
                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False

                    param.grad /= self.world_size
                    dist.reduce(
                        param.grad.data, self._grad_to_rank[index], group=self.process_group, async_op=True,
                    )

                    logging.debug(
                        f"{self.rank}-{index} reducing {id(param)} to {self._grad_to_rank[index]}", flush=True
                    )
                    if self._grad_to_rank[index] != self.global_rank:
                        param.grad = None

            return reduce

        # Go through the parameters, attach the hook
        for i, p in enumerate(filter(lambda x: x.requires_grad, self.base_model.parameters())):
            if p.grad is not None and p.grad.requires_grad:
                raise RuntimeError("ShardedDataParallel only works " "with gradients that don't require grad")

            # Register the hook to the next function in line, so that the hook is fired when this grad
            # has properly been computed
            p_tmp = p.expand_as(p)
            if p_tmp.grad_fn is not None:
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(get_reduce_fn(i))
                self._grad_accs.append(grad_acc)  # keep this function in scope

        logging.info(f"Rank {dist.get_rank(self.process_group)}: All BW hooks are registered")
