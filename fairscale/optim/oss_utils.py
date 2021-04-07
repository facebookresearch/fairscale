# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import chain
import logging
from typing import Any, Dict, List, Optional


def partition_memory(param_groups: List[Dict[str, Any]], world_size: int) -> List[List[Dict[str, Any]]]:
    """ Return partitionned param_groups, trying to minimize the size of each partitiion
    through an eager dispatch to the smallest bucket
    """

    partition_parameters: List[List[Dict[str, Any]]] = [list() for _ in range(world_size)]
    sizes = [0] * world_size
    for param_group in param_groups:
        param_lists: List[List] = [list() for _ in range(world_size)]
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
            partition_parameters[rank].append(param_group_rank)

    return partition_parameters


def partition_keep_ordering(
    param_groups: List[Dict[str, Any]], world_size: int, order_reference: Optional[List[int]]
) -> List[List[Dict[str, Any]]]:

    """ Return partitionned param_groups, trying to minimize the size of each partitiion
    while keeping a strict position ordering
    """

    assert order_reference is None or sum([len(pg["params"]) for pg in param_groups]) == len(
        order_reference
    ), "The ordering index needs to have the same number of references as there are parameters"

    partition_parameters: List[List[Dict[str, Any]]] = [list() for _ in range(world_size)]
    all_params = list(chain.from_iterable((g["params"] for g in param_groups)))

    if order_reference is None:
        # If no order is provided,
        # we assume that the order to be kept is the one from unrolled param groups
        order_reference = list(range(len(all_params)))

    total_number_params = sum(p.numel() for p in all_params)
    number_parameters_per_shard = total_number_params // world_size

    # Set up the empty param groups
    for param_group in param_groups:

        for rank in range(world_size):
            param_group_rank = copy.copy(param_group)
            param_group_rank["params"] = []
            partition_parameters[rank].append(param_group_rank)

    # Set up a param-to-param_group hash table
    param_to_param_group = {}
    for i, pg in enumerate(param_groups):
        for p in pg["params"]:
            param_to_param_group[p] = i

    # Go through the list of parameters in order
    index = sorted(((v, i) for i, v in enumerate(order_reference)))
    current_shard = 0

    for (_, ip) in index:
        param = all_params[ip]
        i_param_group = param_to_param_group[param]

        # Number of parameters in the current shard
        current_shard_params = sum(p.numel() for pg in partition_parameters[current_shard] for p in pg["params"])

        partition_parameters[current_shard][i_param_group]["params"].append(param)

        # This shard is big enough, point to the next one
        if (
            current_shard_params > 0
            and current_shard_params + param.numel() > number_parameters_per_shard
            and current_shard < world_size - 1
        ):
            current_shard += 1

    # Log: show the different bucket sizes
    for i, pp in enumerate(partition_parameters):
        current_shard_params = sum(p.numel() for pg in pp for p in pg["params"])
        logging.info(f"Shard {i}: {current_shard_params/1e6:.2f}M parameters")

    return partition_parameters
