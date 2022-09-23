# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""These functions are used by FullyShardedDataParallel to help consolidate and shard optimizer states."""
import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast

import torch

from fairscale.nn.misc import FlattenParamsWrapper

if TYPE_CHECKING:
    from fairscale.nn.data_parallel import FullyShardedDataParallel

# This function helps shard a full optimizer state dict
def flatten_optim_state_dict(sd: Dict) -> Dict:
    """Shard a full optimizer state dict (called by FSDP.get_shard_from_optim_state_dict)"""
    param_id_map = sd["param_id_map"]
    # Get a set of local ids, like {0, None, 2}, then we remove None from it.
    local_ids = set(param_id_map.values())
    if None in local_ids:
        local_ids.remove(None)
    if sd["state"]:
        new_state: Dict = {local_id: {} for local_id in local_ids}
        singleton_state: Dict = copy.deepcopy(new_state)
    else:
        new_state = {}
    non_tensor_state = {}

    # Populate `new_state["state"]`. (Assuming sd is sorted)
    for global_id, buffers in sd["state"].items():
        local_id = param_id_map[global_id]
        for buffer_name, p in buffers.items():
            if is_singleton_tensor(p):
                singleton_state[local_id][buffer_name] = p
            elif torch.is_tensor(p):
                if buffer_name not in new_state[local_id]:
                    new_state[local_id][buffer_name] = []
                new_state[local_id][buffer_name].append(p.reshape(-1))
            elif isinstance(p, list):
                singleton_state[local_id][buffer_name] = p
            else:
                non_tensor_state[buffer_name] = p
    # Now combine all tensors in each buffer using torch.cat().
    for local_id, state in new_state.items():
        for buffer_name, tensors in state.items():
            new_state[local_id][buffer_name] = torch.cat(tensors)
        new_state[local_id].update(non_tensor_state)
        new_state[local_id].update(singleton_state[local_id])

    # Now make a new param_groups copy and update it.
    new_sd_pg = copy.deepcopy(sd["param_groups"])
    # add pointers from the `params` dict.
    for pg_id, _ in enumerate(sd["param_groups"]):
        # The values() list may look like [0,0,None,None,2,2]. We use
        # groupby to remove the duplicates and then count the length of
        # resulting iter.
        num_local_params = sum(1 for _ in groupby(param_id_map.values()))
        new_sd_pg[pg_id]["params"] = list(range(num_local_params))

    # update the original sd so that we don't lose extra keys, like loss_scale.
    sd["state"] = new_state
    sd["param_groups"] = new_sd_pg
    # delete extra keys we have added to match the original state.
    del sd["uncollected_local_ids"]
    del sd["param_id_map"]
    return sd


def check_param_counts_before_sharding(full_optim_state_dict: Dict, n_instances: int) -> None:
    n_local_params_in_opt = len(set(full_optim_state_dict["param_id_map"].values()))
    msg = (
        f"Including itself, this model has {n_instances} nested instances. When the optimizer state was saved "
        f"there were {n_local_params_in_opt}"
    )
    stateless = len(full_optim_state_dict["state"]) == 0
    assert stateless or (n_instances == n_local_params_in_opt), msg


# All functions below here help saving the list of optimizer states, one from each rank
# build_unflat_state_dict is the interface used by FSDP
def _extract_non_tensor_state(combined_state: Dict[int, Dict[str, List]], param_id: int) -> Dict:
    non_tensor_state = {}  # This state is like the `step` count in Adam, not a tensor so we dont unpad or cat it.
    for k, v in combined_state[param_id].items():
        if torch.is_tensor(v[0]):
            continue
        elif len(set(v)) == 1:
            non_tensor_state[k] = v[0]
        else:
            raise TypeError(f"Dont know how to consolidate optimizer param {k} with values {v}")
    return non_tensor_state


def _unflatten_optim_state(
    combined_state: Dict[int, Dict],
    instance_list: List["FullyShardedDataParallel"],
    world_pad_info: List[List[List[int]]],
    singleton_state: Dict[int, Dict],
) -> Tuple[Dict[int, Dict], Dict[int, int]]:
    """Convert optimizer state for flattened parameters into original, unflattened ones.

    Args:
        combined_state: all-gathered state with tensors
        instance_list: list of FSDP wrapper object instances
        world_pad_info: [param_id][fsdp_instance_id][bytes_padded_per_rank]
        singleton_state: all-gathered dimensionless tensors

    Returns:
        state: unflattened state dict
        idx_mapping: a mapping from global ID to local ID
    """
    # local ids are the keys in the current state (combined_state), (usually fewer)
    # global ids will be the keys in the unflattened state
    next_global_id = 0  # gets incremented
    pad_info = {id: [s[id][0] for s in world_pad_info] for id in combined_state}
    local_ids = [id for id in sorted(combined_state.keys())]

    # non_tensor_state refers to entries in sd[state][param_id] that are not tensors, like "step".
    # we check that these are identical across workers and then take the first
    non_tensor_state = {id: _extract_non_tensor_state(combined_state, id) for id in combined_state}

    # Local corresponds to flattened, global corresponds to unflattened.
    # Casting needed only for mypy.
    num_global_params: List[int] = []
    for m in instance_list:
        if m.flatten_parameters:
            num_flatten = cast(int, m.num_params_managed)
            num_global_params.append(num_flatten)
        else:
            num_global_params.append(len(m.non_shared_params()))
    global_to_local_id = {}
    for local_id, num_unflat in enumerate(num_global_params):
        for _ in range(num_unflat):
            # Some params could be unused, which means the optimizer
            # hasn't created their state. Therefore, `local_id` obtained
            # by enumerating the params above could be out of the range
            # of keys in `combined_state` above. Here is an example:
            #
            #    global    local    notes
            #    0         0        FC1's weight, first flat buffer
            #    1         0        FC1's bias, first flat buffer
            #    2         None     FC2's weight, no flat state
            #    3         None     FC2's bias, no flat state
            #    4         2        FC3's weight, second flat buffer (but with id 2)
            #    5         2        FC3's bias, second flat buffer (but with id 2)
            global_to_local_id[next_global_id] = local_id if local_id in local_ids else None
            next_global_id += 1
    if not combined_state:
        return {}, global_to_local_id

    # copy non tensor state (like the "step" count) to all global entries
    unflat_state = {i: copy.deepcopy(non_tensor_state[0]) for i in range(sum(num_global_params))}

    # remove the global entries that don't have optim state because pytorch
    # optimizer's state_dict() function returns a state_dict without the missing
    # param, so we shouldn't have things like "1:{}" for missing params.
    for g, l in global_to_local_id.items():
        if l is None:
            del unflat_state[g]

    if non_tensor_state[0].keys() == combined_state[0].keys():
        # Early return if there is no tensors in the state dict.
        return unflat_state, global_to_local_id

    local_to_global: Dict[int, List] = {i: [] for i in local_ids}
    for g, l in global_to_local_id.items():
        if l is not None:
            local_to_global[l].append(g)
    # loop over parameters in state.
    # Tensor state will be padded, concatenated, and restored to original shape with FlattenParamsWrapper.get_views
    # get_views returns multiple tensors, each of which is a new parameter with a new "global" id.
    for local_id in local_ids:
        # undo the work of shard_parameters
        for k, v in combined_state[local_id].items():
            if k in non_tensor_state[local_id]:
                continue
            assert isinstance(v, list), f"got {k}: {v} for {local_id}"
            v_unpad = [t[:-np] if np > 0 else t for t, np in zip(v, pad_info[local_id])]
            flat_buffer = torch.cat(v_unpad)
            if instance_list[local_id].flatten_parameters:
                # Unflatten. Casting needed only for mypy.
                param_views: Iterator = cast(FlattenParamsWrapper, instance_list[local_id]).get_param_views(
                    [flat_buffer]
                )
                for global_id, param_view in zip(sorted(local_to_global[local_id]), param_views):
                    assert k not in unflat_state[global_id], f"already added {k} to {global_id} {local_id}"
                    unflat_state[global_id][k] = param_view
            else:
                # Copy non-flatten state directly.
                assert len(local_to_global[local_id]) == 1, "Only support a single non-flatten parameter"
                global_id = local_to_global[local_id][0]
                unflat_state[global_id][k] = flat_buffer
            unflat_state[global_id].update(singleton_state[local_id])

    return unflat_state, global_to_local_id


def build_unflat_state_dict(
    instance_list: List["FullyShardedDataParallel"],
    world_pad_info: List[List[List[int]]],
    state: Dict[int, Dict[str, List[torch.Tensor]]],
    singleton_state: Dict[int, Dict[str, List[torch.Tensor]]],
    uncollected_opt_state: Dict[int, Dict],
    original_sd: Dict,
) -> Dict:
    """Build an unflattened optimizer state dict given a list of flattened optimizer state dicts
    from each rank. This is only called on rank 0.

    Args:
        instance_list: list of FSDP wrapper objects
        world_pad_info: [param_id][fsdp_instance_id][bytes_padded_per_rank]
        state: all-gathered combined/local/flatten state_dict
        singleton_state: all-gathered singleton_state (dimensionless tensors)
        uncollected_opt_state: non-tensor and not-gathered state
        original_sd: the original rank 0's sd

    Returns:
        dict: an unflattened, nonsharded optimizer state, as if FSDP was not there.
    """
    assert all(len(s) == len(instance_list) for s in world_pad_info)
    assert all(len(s[0]) == 1 for s in world_pad_info)

    # Use uncollected_opt_state to update tensor_state, singleton_state
    for local_id, v in uncollected_opt_state.items():
        assert local_id not in state
        state[local_id] = {buffer_name: [x] for buffer_name, x in v.items() if not is_singleton_tensor(x)}
        singleton_state[local_id] = {buffer_name: [x] for buffer_name, x in v.items() if is_singleton_tensor(x)}
    # local ids are in the current state, global_ids will be in returned state.
    unflat_state, global_to_local_id = _unflatten_optim_state(state, instance_list, world_pad_info, singleton_state)

    # Since there are no tensors in param_groups, deepcopy is fine.
    param_groups = copy.deepcopy(original_sd["param_groups"])
    # Casting needed only for mypy.
    num_params = sum([cast(int, m.num_params_managed) for m in instance_list])
    param_groups[0]["params"] = list(range(num_params))

    # Update the original sd so we don't loss extra state like loss_scale.
    original_sd["state"] = dict(sorted(unflat_state.items()))  # NOTE: this is probably already sorted
    original_sd["param_id_map"] = global_to_local_id
    original_sd["param_groups"] = param_groups
    original_sd["uncollected_local_ids"] = list(uncollected_opt_state.keys())
    return original_sd


def is_singleton_tensor(x: Any) -> bool:
    """Is x a dimensionless tensor?"""
    return torch.is_tensor(x) and x.dim() == 0
