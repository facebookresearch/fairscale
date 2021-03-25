# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""These functions are used by FullyShardedDataParallel to help consolidate and shard optimizer states."""
import copy
from typing import Dict, Generator, List, Tuple

import torch


# This function helps shard a full optimizer state dict
def flatten_optim_state_dict(sd: Dict) -> Dict:
    """Shard a full optimizer state dict (called by FSDP.get_shard_from_optim_state_dict)"""
    param_id_map = sd["param_id_map"]
    num_local_params = len(set(param_id_map.values()))
    if sd["state"]:
        new_state: Dict = {local_id: {} for local_id in range(num_local_params)}
    else:
        new_state = {}
    non_tensor_state = {}

    # Populate `new_state["state"]`. (Assuming sd is sorted)
    for expanded_pid, buffers in sd["state"].items():
        consolidated_pid = param_id_map[expanded_pid]
        for buffer_name, p in buffers.items():
            if torch.is_tensor(p):
                if buffer_name not in new_state[consolidated_pid]:
                    new_state[consolidated_pid][buffer_name] = []
                new_state[consolidated_pid][buffer_name].append(p.reshape(-1))
            else:
                non_tensor_state[buffer_name] = p

    # Now combine all tensors in each buffer using torch.cat().
    for consolidated_pid, state in new_state.items():
        for buffer_name, tensors in state.items():
            new_state[consolidated_pid][buffer_name] = torch.cat(tensors)
        new_state[consolidated_pid].update(non_tensor_state)
    new_sd = {"state": new_state, "param_groups": sd["param_groups"]}

    # add pointers from the `params` dict.
    for pg_id, _ in enumerate(sd["param_groups"]):
        # TODO: this list could be huge. Can we avoid materializing?
        new_sd["param_groups"][pg_id]["params"] = list(range(num_local_params))

    return new_sd


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


def _combine_state(states: List[Dict]) -> Dict[int, Dict]:
    combined_state = states[0]
    for param_id in combined_state:
        combined_state[param_id] = {k: [v] for k, v in combined_state[param_id].items()}
    if len(states) == 1:
        return combined_state

    for rank, s in enumerate(states[1:]):
        for param_id, param_state in s.items():
            for k, tensor in param_state.items():
                combined_state[param_id][k].append(tensor)
    return combined_state


def _unflatten_optim_state(
    combined_state: Dict[int, Dict], instance_list: List[torch.nn.Module], world_pad_info: List[List[List[int]]],
) -> Tuple[Dict[int, Dict], Dict[int, int]]:
    # local ids are the keys in the current state (combined_state), (usually fewer)
    # global ids will be the keys in the unflattened state
    next_global_id = 0  # gets incremented
    pad_info = {id: [s[id][0] for s in world_pad_info] for id in combined_state}
    local_ids = [id for id in sorted(combined_state.keys())]

    # non_tensor_state refers to entries in sd[state][param_id] that are not tensors, like "step".
    # we check that these are identical across workers and then take the first
    non_tensor_state = [_extract_non_tensor_state(combined_state, id) for id in combined_state]

    # local corresponds to flattened, global corresponds to unflattened
    num_unflat_params = [len(m._param_numels) for m in instance_list]  # type: ignore
    global_to_local_id = {}
    for local_id, num_unflat in enumerate(num_unflat_params):
        for _ in range(num_unflat):
            global_to_local_id[next_global_id] = local_id
            next_global_id += 1
    if not combined_state:
        return {}, global_to_local_id

    # If the constant state is the same as the combined state,  copy it N times, no unflattening needed.
    unflat_state = {i: copy.deepcopy(non_tensor_state[0]) for i in range(sum(num_unflat_params))}
    if non_tensor_state[0].keys() == combined_state[0].keys():
        return unflat_state, global_to_local_id

    local_to_global: Dict[int, List] = {i: [] for i in local_ids}
    for g, l in global_to_local_id.items():
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
            param_views: Generator = instance_list[local_id].get_param_views(flat_buffer)  # type: ignore
            for global_id, param_view in zip(sorted(local_to_global[local_id]), param_views):
                assert k not in unflat_state[global_id], f"already added {k} to {global_id} {local_id}"
                unflat_state[global_id][k] = param_view

    return unflat_state, global_to_local_id


def build_unflat_state_dict(instance_list: List[torch.nn.Module], world_optim_states: List[Dict]) -> Dict:
    """Build an unflattened optimizer state dict given a list of flattened optimizer state dicts from each rank."""
    world_pad_info: List[List[List[int]]] = [s.pop("num_padded") for s in world_optim_states]
    assert all(len(s) == len(instance_list) for s in world_pad_info)
    assert all(len(s[0]) == 1 for s in world_pad_info)
    param_groups = copy.deepcopy(world_optim_states[0]["param_groups"])
    assert len(param_groups) == 1

    # Aggregate from a list of dictionaries to a dictionary of lists
    combined_state = _combine_state([x["state"] for x in world_optim_states])
    del world_optim_states

    # local ids are in the current state, global_ids will be in returned state.
    unflat_state, global_to_local_id = _unflatten_optim_state(combined_state, instance_list, world_pad_info)
    num_params = sum([len(m._param_numels) for m in instance_list])  # type: ignore
    param_groups[0]["params"] = list(range(num_params))  # This could be a large list. #TODO: is it essential
    return {
        "state": dict(sorted(unflat_state.items())),  # NOTE: this is probably already sorted
        "param_id_map": global_to_local_id,
        "param_groups": param_groups,
    }
