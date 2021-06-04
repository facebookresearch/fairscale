# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

import torch
import torch.fx
from torch.fx.node import Node


def _get_count(param_count: Dict, node_name: str) -> int:
    """Identify different mutations of a given node name."""
    # TODO(anj): This is not very stable since it is possible that the name
    # may not be in the same format. Is there another way to identify nodes
    # in a graph?
    if node_name in param_count:
        return param_count[node_name]
    elif node_name.split("_")[0] in param_count:
        return param_count[node_name.split("_")[0]]
    else:
        raise RuntimeError(f"Unable to find match between param {param_count} and node {node_name}")


def _create_shard_to_param_count(param_count: Dict, node_name_to_shard_id: Dict) -> Dict:
    """Utility to create a map from shard id to param count using existing state."""

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


def _split_nodes(model: torch.nn.Module, shard_count: int = 3) -> Dict:
    """Utility used to trace a graph and identify shard cutpoints."""

    node_name_to_shard_id: Dict[str, int] = {}
    shard_id = 0
    nodes_so_far = []
    param_count: Dict[str, int] = {}
    shard_to_param_count = {}

    traced_graph_module = torch.fx.symbolic_trace(model)

    # Find the total number of params in the model and
    # the number of params per shard we are aiming for.
    for name, module in model.named_modules():
        if "." in name:
            continue
        param_count[name] = sum([x.numel() for x in module.parameters()])

    logging.info(f"Total number of params are {param_count['']}")
    per_shard_param = param_count[""] // shard_count
    logging.info(f"Per shard param count {per_shard_param}")

    for node in traced_graph_module.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
        elif node.op in ["get_attr", "call_function", "call_method", "call_module"]:

            min_shard_id = shard_id
            min_node_name = ""
            # For each of the args of a given node, find the arg that is not the
            # last node we traversed. This is to help us find skip connections
            # across shards.
            for arg in node.args:
                # If the node has args that are inputs to the forward function, they
                # may not have explicit names.
                if not hasattr(arg, "name"):
                    continue

                if arg.name in node_name_to_shard_id and arg.name != nodes_so_far[-1]:
                    if node_name_to_shard_id[arg.name] < min_shard_id:
                        min_shard_id = node_name_to_shard_id[arg.name]
                        min_node_name = arg.name

            # If there is an input that is not from the previous shard,
            # we collapse all the shards in between to be part of 1 shard.
            # and update the param count per shard accordingly.
            if min_shard_id < shard_id:
                for node_name in reversed(nodes_so_far):
                    node_name_to_shard_id[node_name] = min_shard_id
                    if node_name == min_node_name:
                        break
                shard_id = min_shard_id
                # TODO(anj-s): Find a way to raise an error early if this can cause OOM errors.
                shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)

            # Update state that is tracking node -> shard id and shard id -> param count.
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            # TODO(anj): This could just be an update, we don't need to recreate the map.
            shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)
            # If we have gone over the number of params per shard count that we want to
            # achieve, we should add a new shard.
            # The shard_id may not have been updated in the map if we are at a node that does not
            # have params.
            if shard_id in shard_to_param_count and shard_to_param_count[shard_id] > per_shard_param:
                shard_id += 1
        elif node.op == "output":
            break
    return node_name_to_shard_id


def shard_model(model: torch.nn.Module, shard_count: int = 3) -> List[torch.fx.GraphModule]:
    """Utility used to shard a model using torch.fx.

    This function traces the model twice in an attempt to identify the
    right cutpoints and then shard the model. In the first pass we calculate
    the number of parameters as we are tracing the graph and mark nodes at 
    which we might want to create a new module. In the second pass we 
    modify the graph by inserting placeholders and output nodes to essentially
    shard the graph.

    We don't support skip connections between shards. This means that all 
    input and output is self contained within a given shard. A node from
    shard 1 cannot be an input to a node from shard 3. We expect all inputs
    to a given shard to be coming from the last node in the previous shard.
    This means that we may not be able to shard models by the specified
    `shard_count` mentioned by the user. 

    Args:
        model (nn.Module): Model to be sharded as specified by the device count.

        shard_count (int): Number of shards that we want to split the model into.

    """
    module_list: List[torch.fx.GraphModule] = []
    num_graphs = 0
    new_graph = torch.fx.Graph()  # type: ignore
    env: Dict[str, Node] = {}
    new_input_node = None
    # This is the first pass where we attempt to get a map of where
    # we need to insert placeholder and output nodes.
    node_name_to_shard_id = _split_nodes(model, shard_count=shard_count)

    traced_graph_module = torch.fx.symbolic_trace(model)

    # dummy value which indicates that this is the first node.
    prev_shard_id = 1000
    prev_node = None
    for node in traced_graph_module.graph.nodes:
        # If the current node is in the next shard, we insert an output node.
        # A new graph is created and a placeholder is added for the next shard.
        if node.name in node_name_to_shard_id and prev_shard_id < node_name_to_shard_id[node.name]:
            assert prev_node, "prev_node cannot be None"

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
            # Account for a placeholder in the new graph.
            node.args = (new_input_node,)
            new_input_node = None
        if node.op in ["placeholder", "get_attr", "call_function", "call_method", "call_module"]:
            # Copy the nodes from the existing graph to the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        elif node.op == "output":
            # If this is the last node, we should add an output
            # node and add the last graph to the list.
            assert prev_node, "prev_node cannot be None"

            with new_graph.inserting_after(prev_node):
                new_graph.output(env[prev_node.name])
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break
        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]

    return module_list
