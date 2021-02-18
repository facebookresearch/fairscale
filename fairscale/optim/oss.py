# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
from itertools import chain
import logging
from math import inf
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import SGD, Optimizer

from .utils import broadcast_object, recursive_copy_to_device

__all__ = ["OSS"]

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
        broadcast_buffer_size (int):
            (deprecated) used to cap the size of the broadcast buffers, not being used anymore.


    .. warning: the communication patterns that OSS use depend on the "trainability" graph,
        meaning that all the parameters which `require_grad` are handled differently. This is
        not reevaluated at every step, please use `refresh_trainable()` if your model changed
        (freeze or unfreeze for instance).
        If used with :class:<fairscale.nn.ShardedDDP> then an automatic change detection is possible,
        via the `auto_refresh_trainable` parameter.
    """

    #: The optimizer used for a given shard
    optim: Optimizer

    in_super_constructor: bool

    def __init__(
        self,
        params: _params_t,
        optim: Type[Optimizer] = SGD,
        group: Optional[Any] = None,
        broadcast_buffer_size: int = -1,
        **default: Any,
    ):

        # Hold all the model params in the root .param_groups
        self.in_super_constructor = True
        super().__init__(params, default)
        self.in_super_constructor = False

        # Partition information. lazy evaluation, computed when requested
        self._per_device_params: Dict[torch.device, List[List[Parameter]]] = OrderedDict()  # device, rank, params
        self._param_rank: Dict[torch.Tensor, int] = {}
        self._partition_parameters: List[List[dict]] = []
        self._index_to_param: Dict[int, torch.Tensor] = {}
        self._param_to_index: Dict[int, int] = {}
        self._local_params: Optional[List[torch.Tensor]] = None

        # Default empty values + immutables
        self._optim_defaults = default
        self._optim_constructor = optim

        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.global_rank = self.get_global_rank(self.group, self.rank)
        self.buckets: Dict[torch.device, List[torch.Tensor]] = {}

        self._all_states: List[Dict[str, Any]] = []  # Optional consolidated optimizer state
        self._default_device = torch.device("cpu")

        # Setup everything which is related to the parameters to be trained
        # (partition and optimizer for the shard)
        self.refresh_trainable()

    # Partition helpers
    def partition_parameters(self) -> List[List[dict]]:
        """Partitions parameters across distributed data parallel ranks.

        Returns a list of param_groups (which is a list of dict) where each
        element of the list contains the param_groups for a rank. Element 0
        corresponds to rank 0, etc. We need all the ranks for the broadcast
        inside step().
        """
        if len(self._partition_parameters) == 0:
            self._partition_parameters = [list() for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param_group in self.param_groups:
                param_lists: List[List] = [list() for _ in range(self.world_size)]
                for param in param_group["params"]:
                    # Add this param to rank with smallest size.
                    rank = sizes.index(min(sizes))
                    param_lists[rank].append(param)

                    # We're partitioning the optimizer state,
                    # so trainable parameters are the ones which really count
                    if param.requires_grad:
                        sizes[rank] += param.numel()
                    else:
                        # Spread frozen params on a per-tensor basis
                        # Mostly useful for balance partitions for fine tuning for instance
                        # Not required strictly speaking
                        sizes[rank] += 1

                for rank, params in enumerate(param_lists):
                    param_group_rank = copy.copy(param_group)
                    param_group_rank["params"] = params
                    self._partition_parameters[rank].append(param_group_rank)

        return self._partition_parameters

    @property
    def local_params(self) -> List[torch.Tensor]:
        """ Iterable which goes through the parameters that this rank owns
        """
        if self._local_params is None:
            self._local_params = list(
                chain(
                    *[
                        list(filter(lambda x: x.grad is not None, device_params[self.rank]))
                        for device_params in self.per_device_params.values()
                    ]
                )
            )

        # Make sure that the iterator is not consumed, only expose a copy
        return self._local_params

    @property
    def index_to_param(self) -> Dict[int, torch.Tensor]:
        """ Hash table in between parameter indices in the global optimizer scheme, and the actual params
        """
        if len(self._index_to_param) == 0:
            self._index_to_param = {i: p for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))}

        return self._index_to_param

    @property
    def param_to_index(self) -> Dict[int, int]:
        """ Hash table in between parameter indices in the global optimizer scheme, and the actual params
        """
        if len(self._param_to_index) == 0:
            self._param_to_index = {id(p): i for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))}

        return self._param_to_index

    @property
    def per_device_params(self) -> Dict[torch.device, List[List[Parameter]]]:
        """Sorted list of all the params, first per device then per rank.

        Within a list params are sorted per number of elements to allow for an easy bucketing.
        """
        if len(self._per_device_params) == 0:
            # Go through all params, log them per device
            # The ordering is important here, needs to be the same on all ranks
            # So that ulterior broadcast calls are matching
            for param_group in self.param_groups:
                for param in param_group["params"]:
                    device = param.device
                    if self._per_device_params.get(device) is None:
                        self._per_device_params[device] = [[] for _ in range(self.world_size)]
                    self._per_device_params[device][self.param_to_rank[param]] += [param]

            # Sort param_lists by size
            for device in self._per_device_params.keys():
                for rank_params in self._per_device_params[device]:
                    rank_params.sort(key=lambda x: x.numel())

        return self._per_device_params

    @property
    def param_to_rank(self) -> Dict[torch.Tensor, int]:
        """param to data parallel rank"""
        if len(self._param_rank) == 0:
            for rank, param_groups in enumerate(self.partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_rank[param] = rank

            logging.debug("ZeRO: Parameters dispatched to ranks %s " % list(self._param_rank.values()))

        return self._param_rank

    # NOTE(msb) We add a kwargs in order to support Optimizer sub-classes that support extra kwargs.
    # For example, the apex library contains fused optimizers with a step that supports extra kwargs.
    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs: Any) -> Optional[float]:
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note: Any extra parameter is passed to the base optimizer as-is"""

        # Sync oss param_groups attributes in case they've been updated by a scheduler.
        OSS._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only:
        if closure is not None:
            loss = self.optim.step(closure=closure, **kwargs)  # type: ignore
        else:
            loss = self.optim.step(**kwargs)

        # Sync all the updated shards in between the ranks
        self._broadcast_params()

        # Sync hypothethical new results from the wrapped optimizer to the exposed param_groups
        OSS._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        filter_params_fn: Callable[[Any], Any] = None,
    ) -> torch.Tensor:
        """
        Clip all gradients at this point in time. The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note: This is analogous to `torch.nn.utils.clip_grad_norm_` but handles the partitioning and multiple devices per rank
            under the hood. The default torch util is not applicable here, because each rank only has a partial view of all the grads
            in the model, so calling it in the OSS context would lead to different scaling being applied per subset of model parameters

        .. warning: This needs to be called on all ranks, since synchronization primitives will be used

        """

        # Compute the max norm for this shards's worth of gradients
        max_norm = float(max_norm)
        norm_type = float(norm_type)

        # Option to filter parameters from the grad_norm calculation. This is useful for model parallelism.
        # To avoid double counting, only consider parameters on rank zero + anything marked 'model_parallel'
        # 'model_parallel' flag is set in Megatron-LM:
        # https://github.com/NVIDIA/Megatron-LM/blob/19301985dd31c8b612095cbad15bd903e8ddd497/megatron/mpu/layers.py#L54
        local_params = filter_params_fn(self.local_params) if filter_params_fn is not None else self.local_params

        # Compute the norm on this grad set,
        # then sync all the norms from all ranks
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(self._default_device) for p in local_params)
            # all reduce over data parallel and model parallel workers
            dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=dist.group.WORLD)
        else:
            local_norm = torch.norm(
                input=torch.stack([torch.norm(input=p.grad.detach(), p=norm_type, dtype=torch.float32).to(self._default_device) for p in local_params]),  # type: ignore
                p=norm_type,
            )

            # local norm result can be accumulated with the remote ones if put to the right power
            # n_i = sum_rank(a^p)^1/p
            # -> n_total = all_reduce(n_i^p)^(1/p) = sum_i(n_i^p)^1/p = sum_i(sum_rank(a^p))^1/p
            # all reduce over data parallel and model parallel workers
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm)
            total_norm = total_norm ** (1.0 / norm_type)

        clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-6)
        if clip_coef < 1:
            for device, device_params in self.per_device_params.items():
                for p in filter(lambda x: x.grad is not None, device_params[self.rank]):
                    p.grad.detach().mul_(clip_coef.to(device))  # type: ignore

        return total_norm

    # State dict interfaces
    def consolidate_state_dict(self, recipient_rank: int = 0) -> None:
        """Update the consolidated state_dict list, one per rank.

        .. warning: This needs to be called on all replicas"""

        # Sync lr and other attributes in case its been updated
        OSS._sync_param_groups(self.param_groups, self.optim.param_groups)

        if self.rank == recipient_rank:
            # Pull the sharded state from all the other replicas
            # Store all the states in order, rank by rank
            logging.debug("Pulling the sharded optimizer state from all replicas")
            self._all_states = self._collect_sharded_states()
        else:
            # Acknowledge broadcasts, and send this rank's shard when needed
            self._broadcast_state_dict()

    def local_state_dict(self) -> dict:
        """ .. deprecated:: 0.1.5

        Returns this rank's state_dict as a :class:`dict` which contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.

        * param_groups - a dict containing all parameter groups

        .. warning: This does not represent the optimizer state dict, only a shard.
        """
        return self.optim.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        """Return the last known global optimizer state. The returned state is compatible with Pytorch, in that the
        sharded properties are not exposed. It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.

        * param_groups - a dict containing all parameter groups

        .. warning:
            If the state has not been consolidated, this returns a shard's worth, not the global state.

        .. warning:
            Returning the global state is limited to the replica which was responsible for the consolidation.
            The state may also not be up to date, depending on when `consolidate_state_dict` was last called.
        """

        if len(self._all_states) == 0:
            raise RuntimeError(
                "Optimizer state has not been consolidated on this rank. \
                Please call `consolidate_state_dict()` on all ranks beforehand if you meant to save the global state"
            )

        # Unify the shard states and the state that pytorch would expect, given the model.
        # Indexation needs several redirections, since each shard only knows a limited scope of the model
        # - get the pytorch compliant parameter indexing
        state_dict = super().state_dict()

        # - go through the per-shard states, which are all indexed locally
        for rank, s in enumerate(self._all_states):
            # -- match the local indexing and the global partition, update the corresponding saved state globally
            for local_pg, global_pg in zip(s["param_groups"], self.partition_parameters()[rank]):
                local_index_to_param_id = {
                    i_param: id(global_pg["params"][i]) for i, i_param in enumerate(local_pg["params"])
                }

                for local_param_index in local_pg["params"]:
                    # Update the state, if any
                    if local_param_index in s["state"].keys():
                        global_id = self.param_to_index[local_index_to_param_id[local_param_index]]
                        state_dict["state"][global_id] = s["state"][local_param_index]

        # Make sure that the parameters are sorted in the state, as expected
        state_dict["state"] = dict(sorted(state_dict["state"].items()))
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the global parameter groups as well as the shard.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`
        """

        # NOTE: PyTorch 1.5 does not index linearly but with the id(params) at saving time
        # we work around that here by using the fact that the params are ordered as in the param_groups
        pytorch15_index_redirect = {k: i for i, k in enumerate(state_dict["state"].keys())}

        for key, value in state_dict["state"].items():
            param = self.index_to_param[pytorch15_index_redirect[key]]

            # Populate the sharded optimizer state on the fly
            if self.param_to_rank[param] != self.rank:
                state_dict["state"][key] = None
            else:
                self.optim.state[param] = recursive_copy_to_device(value, non_blocking=True, device=param.device)

        super().load_state_dict(state_dict)

        # Sync with the optimizer param groups
        OSS._sync_param_groups(state_dict["param_groups"], self.param_groups)
        OSS._sync_param_groups(self.param_groups, self.optim.param_groups)

    def refresh_trainable(self) -> None:
        """ Updates the partitioning and communication patterns if the trainability (`requires_grad`)
        of some parameters changed.
        """

        # Create the optim which will work on the param shard
        if not hasattr(self, "optim"):
            self._clear_cache()
            self._default_device = list(self.per_device_params.keys())[0]
            self.optim = self._optim_constructor(self.partition_parameters()[self.rank], **self._optim_defaults)
            OSS._sync_param_groups(self.optim.param_groups, self.param_groups)

        self._setup_flat_buffers()

    def _broadcast_state_dict(self) -> None:
        """Broadcast this rank's state shard, discard others"""

        # Tensor cannot be really empty, even if its size is meaningless
        dummy_sync_tensor = torch.tensor([1], device=self._default_device)

        for rank in range(self.world_size):
            if rank == self.rank:
                # Send the state to the reference replica
                logging.debug(
                    "Sending the sharded optimizer state to the reference replica from rank %s", rank,
                )
                # legacy compatibility for old torch versions
                broadcast_object(
                    self.local_state_dict(),
                    src_rank=self.global_rank,
                    group=self.group,
                    dist_device=self._default_device,
                )
            else:
                global_rank = self.get_global_rank(self.group, rank)

                # Discard this tensor/rank, broadcast necessary for syncing and because NCCL does not support gather
                broadcast_object(
                    torch.tensor([dummy_sync_tensor], dtype=torch.uint8, device=self._default_device),
                    src_rank=global_rank,
                    group=self.group,
                    dist_device=self._default_device,
                )

    def _collect_sharded_states(self) -> List[Dict[str, Any]]:
        """Collect all the state shards, in CPU memory."""
        all_states = []

        for rank in range(self.world_size):
            if rank == self.rank:
                logging.debug("Saving self state")
                all_states.append(
                    recursive_copy_to_device(self.optim.state_dict(), non_blocking=True, device=torch.device("cpu"))
                )

                # Sync with other replicas
                broadcast_object(
                    torch.tensor([0], dtype=torch.uint8, device=self._default_device),
                    src_rank=self.global_rank,
                    group=self.group,
                    dist_device=self._default_device,
                )
            else:
                # Fetch the optim state from the other replicas
                global_rank = self.get_global_rank(self.group, rank)
                replica_state = broadcast_object(
                    torch.tensor([0], dtype=torch.uint8, device=self._default_device),
                    src_rank=global_rank,
                    group=self.group,
                    dist_device=self._default_device,
                )

                all_states.append(
                    recursive_copy_to_device(replica_state, non_blocking=True, device=torch.device("cpu"))
                )

                logging.debug("State from rank %s received", rank)

        return all_states

    def add_param_group(self, param_group: dict) -> None:
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options

        .. warning: This handles updating the shards on all partitions, but needs to be called on all ranks.
        """

        super().add_param_group(param_group)
        if not self.in_super_constructor:
            # Force a re-partitioning
            self._clear_cache()

            # Update the partition
            param_groups = self.partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            self._setup_flat_buffers()

    def _clear_cache(self) -> None:
        self._partition_parameters.clear()
        self._per_device_params.clear()
        self._param_rank.clear()
        self._index_to_param.clear()
        self._param_to_index.clear()
        self._local_params = None

    @staticmethod
    def get_global_rank(group: Any, rank: int) -> int:
        if group is dist.group.WORLD:
            return rank
        else:
            global_rank = dist.distributed_c10d._get_global_rank(group, rank)
        return global_rank

    @staticmethod
    def _sync_param_groups(source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]) -> None:
        """Sync learning rate and other optimizer attributes (needed to support schedulers)."""

        for source_group, destination_group in zip(source, destination):
            # Sync everything but the parameters
            for k in filter(lambda x: x != "params", source_group.keys()):
                destination_group[k] = source_group[k]

    @torch.no_grad()
    def _broadcast_params(self) -> None:
        """Helper function to broadcast all the parameters from a given device"""

        last_work_handle = None  # Work handles are consumed within this scope, no callback

        for device in self.buckets.keys():
            for src_rank, bucket in enumerate(self.buckets[device]):
                global_src_rank = self.get_global_rank(self.group, src_rank)
                last_work_handle = dist.broadcast(tensor=bucket, src=global_src_rank, group=self.group, async_op=True)

        # Only check on the last handle, they're all inlined on the same CUDA stream
        if last_work_handle:
            last_work_handle.wait()

    def _setup_flat_buffers(self) -> None:
        """Make all params which are on the same device and tied to the same rank views of a single buffer.
        This is used at construction time, and anytime parameter trainability is changed (frozen or unfrozen) and
        `refresh_trainability` is called.
        """

        for device, per_rank_params in self.per_device_params.items():
            # Only wipe the existing buckets if there are none
            # (could be that this is called twice, when trainability changes)
            if device not in self.buckets.keys():
                self.buckets[device] = []

            # Make parameters a view of the bucket
            for dst_rank, params in enumerate(per_rank_params):
                if len(params) > 0:

                    # Clone the non-trainable params, if in a bucket it will get destroyed
                    for param in filter(lambda x: not x.requires_grad, params):
                        param.data = param.data.detach().clone()

                    # Merge all the trainable params in a single bucket
                    trainable_params = list(filter(lambda x: x.requires_grad, params))
                    buffer_size = sum(map(lambda x: x.numel(), trainable_params))
                    bucket = torch.empty(buffer_size, dtype=params[0].dtype, device=device)
                    offset = 0

                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next

                    # Either replace the existing bucket, or create it
                    if len(self.buckets[device]) == dst_rank:
                        self.buckets[device].append(bucket)
                    else:
                        self.buckets[device][dst_rank] = bucket
                else:
                    self.buckets[device].append(torch.zeros(1, device=device))
