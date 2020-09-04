# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import chain
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Tuple

import torch
import torch.distributed as dist
from torch.optim import SGD, Optimizer

from .utils import broadcast_object, recursive_copy_to_device

if TYPE_CHECKING:  # pragma: no cover
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class OSS(Optimizer):
    """Wraps an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>`
    optimizer and shards its state as described by ZeRO_.
    ::
        opt = OSS(params, optim=torch.optim.Adam, lr=0.01)

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    We use a greedy algorithm to pack a number of parameters
    at each rank. Each parameter belongs to a single rank and
    is not divided among rank.

    After each rank completed their parameter update, they broadcast
    the new version of the parameters to all other ranks to synchronize
    the parameters for next round forward/backward computation.

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

    def __init__(self, params: _params_t, optim: Type[Optimizer] = SGD, group: Any = dist.group.WORLD, **defaults: Any):
        # Hold all the model params in the root .param_groups
        self.in_super_constructor = True
        super().__init__(params, defaults)
        self.in_super_constructor = False

        # Build the wrapped optimizer, responsible for a shard of the params
        self.group = group
        self.rank = dist.get_rank(group)
        split_param_groups = self.partition_parameters()
        self.optim = optim(split_param_groups[self.rank], **defaults)

        # Optional consolidated optimizer state
        self._all_states: List[Dict[str, Any]] = []

        # Current device is set by the parameters allocated to this rank
        self._device = split_param_groups[self.rank][0]["params"][0].device

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
                param_group_rank = copy.copy(param_group)
                param_group_rank["params"] = params
                param_groups[rank].append(param_group_rank)
        return param_groups

    # NOTE(msb) We add a kwargs in order to support Optimizer sub-classes that support extra kwargs.
    # For example, the apex library contains fused optimizers with a step that supports extra kwargs.
    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs: Any) -> Optional[float]:
        # Sync lr in case its been update by an LRScheduler.
        self._sync_lr()

        # Run the optimizer step on this shard only
        loss = self.optim.step(closure=closure, **kwargs)  # type: ignore

        # Sync all the states
        for rank, param_groups in enumerate(self.partition_parameters()):
            for param_group in param_groups:
                for param in param_group["params"]:
                    dist.broadcast(tensor=param, src=rank, group=self.group)
        return loss

    def local_state_dict(self) -> dict:
        """ Gets this rank's state_dict. """
        return self.optim.state_dict()

    def consolidate_state_dict(self, recipient_rank: int = 0) -> None:
        """ Update the consolidated state_dict list, one per rank.

        This needs to be called on all replicas """

        # Sync lr in case its been update by an LRScheduler.
        self._sync_lr()

        if self.rank == recipient_rank:
            # Pull the sharded state from all the other replicas
            # Store all the states in order, rank by rank
            logging.debug("Pulling the sharded optimizer state from all replicas")
            self._all_states = self._collect_sharded_states()
        else:
            # Acknowledge broadcasts, and send this rank's shard when needed
            self._broadcast_state_dict()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the last known global optimizer state, which consist of a list of the shards.

        NOTE: This is limited to the replica which was responsible for the consolidation.
        The state may also not be up to date, depending on when `consolidate_state_dict` was last called.
        """

        assert (
            len(self._all_states) > 0
        ), "The optimizer state is not materialized, please call consolidate_state_dict on every replica beforehand"

        # Flatten the state_dict, save the partition
        # The partition indexes the flat param_groups list,
        # and logs the rank <> shard correspondence
        partition: Dict[int, Tuple[int, int]] = {}
        param_groups: List[Dict[Any, Any]] = []

        i_global = 0
        for i, s in enumerate(self._all_states):
            param_groups.extend(s["param_groups"])
            partition[i] = (i_global, i_global + len(s["param_groups"]))
            i_global = partition[i][1]

        return {
            "state": {i: s["state"] for i, s in enumerate(self._all_states)},
            "param_groups": param_groups,
            "partition": partition,
        }

    def load_local_state_dict(self, state_dict: dict) -> None:
        """ Loads this rank's state_dict. """

        self.optim.load_state_dict(state_dict)

        # Workaround PyTorch bug that casts state (https://github.com/pytorch/pytorch/issues/43706)
        # Copied from https://github.com/pytorch/fairseq/blob/v0.9.0/fairseq/optim/fp16_optimizer.py#L251-L268
        groups = self.optim.param_groups
        saved_groups = state_dict["param_groups"]
        id_map = {
            old_id: p
            for old_id, p in zip(chain(*(g["params"] for g in saved_groups)), chain(*(g["params"] for g in groups)))
        }
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                self.optim.state[param] = recursive_copy_to_device(v, non_blocking=True, device=param.device)

        # Restore the global param_groups (the params themselves are already correct)
        for global_group, local_group in zip(self.param_groups, groups):
            for k, v in local_group.items():
                if k != "params":
                    global_group[k] = v

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Restore the global parameter groups as well as the shard """

        # Get this optimizer's param_groups shard
        param_groups = state_dict["param_groups"][
            state_dict["partition"][self.rank][0] : state_dict["partition"][self.rank][1]
        ]
        # Dispatch this rank's state dictionary to the wrapped shard optimizer
        self.load_local_state_dict({"state": state_dict["state"][self.rank], "param_groups": param_groups})

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if not self.in_super_constructor:
            param_groups = self.partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

    def _sync_lr(self) -> None:
        """Sync learning rate (needed to support LRScheduler)."""
        for global_group, local_group in zip(self.param_groups, self.optim.param_groups):
            local_group["lr"] = global_group["lr"]

    def _collect_sharded_states(self) -> List[Dict[str, Any]]:
        """
        Collect all the state shards, in CPU memory.
        """
        empty_buffer = torch.tensor([0], dtype=torch.uint8, device=self._device)
        all_states: List[Dict[str, Any]] = []

        for rank in range(dist.get_world_size(group=self.group)):
            if rank == self.rank:
                logging.debug("Saving self state")
                all_states.append(
                    recursive_copy_to_device(self.local_state_dict(), non_blocking=True, device=torch.device("cpu"))
                )

                # Sync with other replicas
                broadcast_object(empty_buffer, src_rank=rank, group=self.group, dist_device=self._device)
            else:
                # Fetch the optim state from the other replicas
                logging.debug("Receiving state from rank %s ", rank)
                replica_state = broadcast_object(
                    empty_buffer, src_rank=rank, group=self.group, dist_device=self._device
                )

                all_states.append(
                    recursive_copy_to_device(replica_state, non_blocking=True, device=torch.device("cpu"))
                )

                logging.debug("State from rank %s received", rank)

        return all_states

    def _broadcast_state_dict(self) -> None:
        """
        Broadcast this rank's state shard, discard others
        """
        empty_buffer = torch.tensor([0], dtype=torch.uint8, device=self._device)

        for rank in range(dist.get_world_size(group=self.group)):
            if rank == self.rank:
                # Send the state to the reference replica
                logging.debug(
                    "Sending the sharded optimizer state to the reference replica from rank %s", rank,
                )
                broadcast_object(self.local_state_dict(), src_rank=rank, group=self.group, dist_device=self._device)
            else:
                # Discard this tensor/rank, broadcast necessary for syncing
                logging.debug("Discarding broadcast from rank %s", rank)
                broadcast_object(empty_buffer, src_rank=rank, group=self.group, dist_device=self._device)
