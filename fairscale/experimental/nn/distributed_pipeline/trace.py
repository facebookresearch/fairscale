import inspect
import operator
from typing import Dict, List, Optional, Tuple, cast

from torch.distributed.nn import RemoteModule
import torch.fx
import torch.nn as nn

from . import PipelineModulesGraph


class RemoteModuleTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, RemoteModule):
            return True
        return False


class GraphCreator:
    def __init__(self, tracer: RemoteModuleTracer) -> None:
        self.tracer = tracer

    def get_module(self, node: torch.fx.Node) -> Optional[nn.Module]:
        """Given a call_module node, returns the module corresponding to this module call"""
        if node.op != "call_module":
            return None
        module = self.tracer.root
        for t in cast(str, node.target).split("."):
            module = getattr(module, t)
        return module

    def create_graph(self) -> PipelineModulesGraph:
        # node_to_data maps nodes to the data they represent
        node_to_data: Dict[torch.fx.Node, PipelineModulesGraph.DataSourceSpec] = {}
        remote_module_nodes: List[Tuple[torch.fx.Node, RemoteModule]] = []

        for node in self.tracer.graph.nodes:
            if node.op == "call_module":
                module = self.get_module(node)
                assert isinstance(module, RemoteModule)
                node_to_data[node] = module
                remote_module_nodes.append((node, module))
            elif node.target == operator.__getitem__ and node.op == "call_function":
                assert node.args[0] in node_to_data
                d = node_to_data[node.args[0]]
                assert isinstance(d, RemoteModule)
                node_to_data[node] = (d, node.args[1])
            elif node.op == "placeholder":
                arg_names = list(inspect.signature(self.tracer.root.forward).parameters)
                node_to_data[node] = arg_names.index(node.target)
            elif node.op == "output":
                pass
            else:
                assert False, "Invalid node %s" % node

        # Following dict stores cardinality of the output for modules: for each module, it stores None if the output
        # is a simple tensor, and it stores the number of tensors in the the output if it is a tuple.
        module_to_num_outputs: Dict[nn.Module, Optional[int]] = {}
        for node, _ in remote_module_nodes:
            # iterate over inputs to the module.
            for arg in node.args:
                data = node_to_data[arg]
                if isinstance(data, int):
                    continue
                if isinstance(data, RemoteModule):
                    assert module_to_num_outputs.get(data, None) is None
                    module_to_num_outputs[data] = None
                else:
                    module, output_num = data
                    # Here we discovered that the output number "output_num" is used,
                    # so the number of outputs should be at least "output_num+1".
                    if module in module_to_num_outputs:
                        prev_value = module_to_num_outputs[module]
                        assert prev_value is not None
                        module_to_num_outputs[module] = max(prev_value, output_num + 1)
                    else:
                        module_to_num_outputs[module] = output_num + 1

        graph = PipelineModulesGraph()
        for node, module in remote_module_nodes:
            inputs = [node_to_data[arg] for arg in node.args]
            graph.add_layer(module, inputs, module_to_num_outputs.get(module))

        return graph


def _call_trace(tracer: RemoteModuleTracer, module: nn.Module) -> torch.fx.Graph:
    try:
        org_named_modules = RemoteModule.named_modules
        org_named_children = RemoteModule.named_children
        RemoteModule.named_modules = nn.Module.named_modules  # type: ignore
        RemoteModule.named_children = nn.Module.named_children  # type: ignore
        return tracer.trace(module)
    finally:
        RemoteModule.named_modules = org_named_modules  # type: ignore
        RemoteModule.named_children = org_named_children  # type: ignore


def make_graph(module: nn.Module) -> PipelineModulesGraph:
    """
    Creates a PipelineModulesGraph for the module. The module should be traceable by torch.fx.
    Also all operators on tensors should be done by RemoteModule's.
    """
    tracer = RemoteModuleTracer()
    r = _call_trace(tracer, module)
    g = torch.fx.GraphModule(module, r)
    return GraphCreator(tracer).create_graph()
