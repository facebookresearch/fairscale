# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""These functions are used by FullyShardedDataParallel to help consolidate and shard optimizer states."""
import copy
from typing import Any, Dict, Generator, List, Tuple

import torch

# These return keys are used by fairseq. To change, add @sshleifer as a reviewer.
UNFLAT_RETURN_KEYS = {"state", "param_groups", "uncollected_local_ids", "param_id_map"}

# This function helps shard a full optimizer state dict
def flatten_optim_state_dict(sd: Dict) -> Dict:
    """Shard a full optimizer state dict (called by FSDP.get_shard_from_optim_state_dict)"""
    param_id_map = sd["param_id_map"]
    num_local_params = len(set(param_id_map.values()))
    if sd["state"]:
        new_state: Dict = {local_id: {} for local_id in range(num_local_params)}
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
    new_sd = {"state": new_state, "param_groups": copy.deepcopy(sd["param_groups"])}
    for k in sd.keys():  # if there are extra keys, like loss_scale, don't delete them
        if k not in UNFLAT_RETURN_KEYS:
            new_sd[k] = copy.deepcopy(sd[k])

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


def _unflatten_optim_state(
    combined_state: Dict[int, Dict],
    instance_list: List[torch.nn.Module],
    world_pad_info: List[List[List[int]]],
    singleton_state: Dict[int, Dict],
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
    num_global_params = [len(m._param_numels) for m in instance_list]  # type: ignore
    global_to_local_id = {}
    for local_id, num_unflat in enumerate(num_global_params):
        for _ in range(num_unflat):
            global_to_local_id[next_global_id] = local_id
            next_global_id += 1
    if not combined_state:
        return {}, global_to_local_id

    # copy non tensor state to all global entries
    unflat_state = {i: copy.deepcopy(non_tensor_state[0]) for i in range(sum(num_global_params))}

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
                unflat_state[global_id].update(singleton_state[local_id])

    return unflat_state, global_to_local_id


def build_unflat_state_dict(
    instance_list: List[torch.nn.Module],
    world_pad_info: List[List[List[int]]],
    state: Dict[int, Dict[str, List[torch.Tensor]]],
    singleton_state: Dict[int, Dict[str, List[torch.Tensor]]],
    uncollected_opt_state: Dict[int, Dict],
    param_groups: List[Dict],
) -> Dict:
    """Build an unflattened optimizer state dict given a list of flattened optimizer state dicts from each rank."""
    assert all(len(s) == len(instance_list) for s in world_pad_info)
    assert all(len(s[0]) == 1 for s in world_pad_info)

    # Use uncollected_opt_state to update tensor_state, singleton_state
    for local_id, v in uncollected_opt_state.items():
        assert local_id not in state
        state[local_id] = {buffer_name: [x] for buffer_name, x in v.items() if not is_singleton_tensor(x)}
        singleton_state[local_id] = {buffer_name: [x] for buffer_name, x in v.items() if is_singleton_tensor(x)}
    # local ids are in the current state, global_ids will be in returned state.
    unflat_state, global_to_local_id = _unflatten_optim_state(state, instance_list, world_pad_info, singleton_state)
    # Since there are no tensors in param_groups, deepcopy is fine
    param_groups = copy.deepcopy(param_groups)
    num_params = sum([len(m._param_numels) for m in instance_list])  # type: ignore
    param_groups[0]["params"] = list(range(num_params))
    unflat_optim_state_dict = {
        "state": dict(sorted(unflat_state.items())),  # NOTE: this is probably already sorted
        "param_id_map": global_to_local_id,
        "param_groups": param_groups,
        "uncollected_local_ids": list(uncollected_opt_state.keys()),
    }
    assert set(unflat_optim_state_dict.keys()) == UNFLAT_RETURN_KEYS
    return unflat_optim_state_dict


def is_singleton_tensor(x: Any) -> bool:
    """Is x a dimensionless tensor?"""
    return torch.is_tensor(x) and x.dim() == 0
