# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

import torch
import torch.distributed as dist
from torch.optim import SGD, Optimizer

from .utils import broadcast_object, recursive_copy_to_device

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class OSS(Optimizer):
    """Wraps an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>`
    optimizer and shards its state as described by ZeRO_.
    ::
        opt = OSS(params, optim=torch.optim.Adam, lr=0.01)

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`Pipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    Args:
        params (list of tensors):
            parameters to be optimized
    Keyword Args:
        optim (torch.nn.Optimizer):
            optimizer to shard (default: SGD)
        group (group):
            torch.distributed group (default: group.WORLD)
    """

    optim: Optimizer
    in_super_constructor: bool

    def __init__(
        self,
        params: _params_t,
        optim: Type[Optimizer] = SGD,
        group: Any = dist.group.WORLD,
        **defaults: Any
    ):
        self.in_super_constructor = True
        super().__init__(params, defaults)
        self.in_super_constructor = False

        self.group = group
        self.rank = dist.get_rank(group)
        param_groups = self.partition_parameters()
        self.optim = optim(param_groups[self.rank], **defaults)

        # Optional consolidated optimizer state
        self._global_state_dict = []

    def partition_parameters(self) -> List[List[dict]]:
        """Partitions parameters across distributed ranks.

        Returns a list of param_groups (which is a list of dict) where each
        element of the list contains the param_groups for a rank. Element 0
        corresponds to rank 0, etc. We need all the ranks for the broadcast
        inside step().
        """
        world_size = dist.get_world_size(self.group)
        param_groups: List[List] = [list() for _ in range(world_size)]
        sizes = [0] * world_size
        for param_group in self.param_groups:
            param_lists: List[List] = [list() for _ in range(world_size)]
            for param in param_group["params"]:
                # Add this param to rank with smallest size.
                rank = sizes.index(min(sizes))
                param_lists[rank].append(param)
                sizes[rank] += param.numel()
            for rank, params in enumerate(param_lists):
                if len(params) > 0:
                    pg = copy.copy(param_group)
                    pg["params"] = params
                    param_groups[rank].append(pg)
        return param_groups

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = self.optim.step(closure=closure)
        for rank, param_groups in enumerate(self.partition_parameters()):
            for param_group in param_groups:
                for param in param_group["params"]:
                    dist.broadcast(param, rank, group=self.group)
        return loss

    def state_dict(self) -> dict:
        """ Gets this rank's state_dict. """
        return self.optim.state_dict()

    def _collect_state_dict(self) -> List[dict]:
        """
        Collect all the state shards
        """
        empty_buffer = torch.empty([1], dtype=torch.uint8)
        global_optim_state = []
        local_state = self.state_dict()

        if len(local_state["state"]) == 0:
            return []

        for rank in range(dist.get_world_size(group=self.group)):
            if rank == self.rank:
                logging.info("Saving self state")
                global_optim_state.append(
                    recursive_copy_to_device(
                        local_state, non_blocking=True, device=torch.device("cpu")
                    )
                )

                # Sync with other replicas
                broadcast_object(empty_buffer, src_rank=rank)
            else:
                # Reuse the param_groups from this rank, these are shared across replicas
                logging.info("Receiving state from rank %s ", rank)
                replica_state = {
                    "state": broadcast_object(empty_buffer, src_rank=rank),
                    "param_groups": local_state["param_groups"],
                }

                # Fetch from the other replicas
                global_optim_state.append(
                    recursive_copy_to_device(
                        replica_state, non_blocking=True, device=torch.device("cpu")
                    )
                )
                logging.info("State from rank %s received", rank)

        return global_optim_state

    def _broadcast_state_dict(self) -> None:
        """
        Broadcast this rank's state shard, discard others
        """
        empty_buffer = torch.empty([1], dtype=torch.uint8)
        local_state = self.state_dict()

        if len(local_state["state"]) == 0:
            return

        for rank in range(dist.get_world_size(group=self.group)):
            if rank == self.rank:
                # Send the state to the reference replica
                logging.info(
                    "Sending the sharded SGD state to the reference replica from rank %s",
                    rank,
                )
                broadcast_object(local_state["state"], src_rank=rank)
            else:
                # Discard this tensor/rank, broadcast necessary for syncing
                logging.info("Discarding broadcast from rank %s", rank)
                broadcast_object(empty_buffer, src_rank=rank)

    def consolidate_state_dict(self, recipient_rank: int = 0) -> List[dict]:
        """ Update the consolidated state_dict list, one per rank.

        This needs to be called on all replicas """

        if self.rank == recipient_rank:
            # Pull the sharded state from all the other replicas
            # Store all the states in order, rank by rank
            logging.info("Pulling the sharded SGD state from all replicas")
            self._global_state_dict = self._collect_state_dict()
        else:
            # Acknowledge broadcasts, and send this rank's shard when needed
            self._broadcast_state_dict()

    @property
    def global_state_dict(self):
        """
        Return the last known global optimizer state, which consist of a list of the shards.

        NOTE: This is limited to the replica which was responsible for the consolidation.
        The state may also not be up to date, depending on when `consolidate_state_dict` was last called
        """

        assert (
            len(self._global_state_dict) > 0
        ), "The optimizer state is not materialized, please call consolidate_state_dict on every replica beforehand"

        return self._global_state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """ Loads this rank's state_dict. """
        self.optim.load_state_dict(state_dict)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if not self.in_super_constructor:
            param_groups = self.partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])
