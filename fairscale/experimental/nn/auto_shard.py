# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Any, Dict, List

import torch
import torch.fx
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node


def _get_count(param_count: Dict, node_name: Any) -> int:
    if node_name in param_count:
        return param_count[node_name]
    elif node_name.split("_")[0] in param_count:
        return param_count[node_name.split("_")[0]]
    else:
        raise RuntimeError(f"Unable to find match between param {param_count} and node {node_name}")


def _create_shard_to_param_count(param_count: Dict, node_name_to_shard_id: Dict) -> Dict:
    shard_to_param_count: Dict[int, int] = {}
    for node_name in node_name_to_shard_id.keys():
        try:
            count = _get_count(param_count, node_name)
        except RuntimeError:
            continue
        if node_name_to_shard_id[node_name] in shard_to_param_count:
            shard_to_param_count[node_name_to_shard_id[node_name]] += count
        else:
            shard_to_param_count[node_name_to_shard_id[node_name]] = count
    return shard_to_param_count


def _split_nodes(model: Any, shard_count: int = 3) -> Dict:

    node_name_to_shard_id: Dict[str, int] = {}
    traced_graph_module = torch.fx.symbolic_trace(model)

    shard_id = 0
    nodes_so_far = []
    param_count: Dict[str, int] = {}
    shard_to_param_count = {}

    for named_mods in model.named_modules():
        sum = 0
        for x in named_mods[1].parameters():
            mul_dims = math.prod(x.size())
            sum += mul_dims

        name = named_mods[0].split(".")[0]
        if name in param_count:
            param_count[name] += sum
        else:
            param_count[name] = sum

    logging.info(f"Total params are {param_count['']}")
    per_shard_param = param_count[""] // shard_count
    logging.info(f"Per shard param count {per_shard_param}")

    for node in traced_graph_module.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
        elif node.op in ["get_attr", "call_function", "call_method", "call_module"]:

            min_shard_id = shard_id
            for arg in node.args:
                try:
                    test = arg.name
                except AttributeError:
                    continue

                if arg.name in node_name_to_shard_id and arg.name != nodes_so_far[-1]:
                    min_shard_id = min(min_shard_id, node_name_to_shard_id[arg.name])

            if min_shard_id < shard_id:
                for node_name in reversed(nodes_so_far):
                    node_name_to_shard_id[node_name] = min_shard_id
                shard_id = min_shard_id
                shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)

            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)
            if shard_id in shard_to_param_count and shard_to_param_count[shard_id] > per_shard_param:
                shard_id += 1
        elif node.op == "output":
            break
    return node_name_to_shard_id


def shard_model(model: Any, shard_count: int = 3) -> List[GraphModule]:
    """Utility used to shard a model using torch.fx."""
    module_list = []
    num_graphs = 0
    new_graph = torch.fx.Graph()
    env: Dict[str, Node] = {}
    new_input_node = None
    node_name_to_shard_id = _split_nodes(model, shard_count=shard_count)

    traced_graph_module = torch.fx.symbolic_trace(model)

    prev_shard_id = 1000
    prev_node = None
    for node in traced_graph_module.graph.nodes:
        if node.name in node_name_to_shard_id and prev_shard_id < node_name_to_shard_id[node.name]:
            with new_graph.inserting_after(prev_node):
                new_graph.output(env[prev_node.name])
            num_graphs += 1
            module_list.append(torch.fx.GraphModule(model, new_graph))
            new_graph = torch.fx.Graph()
            node_name = "placeholder" + str(num_graphs)
            pl_node = new_graph.create_node("placeholder", node_name)
            env[node_name] = pl_node
            new_input_node = pl_node

        if new_input_node is not None:
            node.args = (new_input_node,)
            new_input_node = None
        if node.op in ["placeholder", "get_attr", "call_function", "call_method", "call_module"]:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        elif node.op == "output":
            with new_graph.inserting_after(prev_node):
                new_graph.output(env[prev_node.name])
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break
        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]

    return module_list
