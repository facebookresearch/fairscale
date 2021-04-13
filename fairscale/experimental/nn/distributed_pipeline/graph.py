# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from torch import Tensor, nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule

from .data import DataConsumer


class MultiInputSequential(nn.Module):
    """A variation of nn.Sequential, that allows the first module in the sequence accepts
        multiple inputs. To be used internally by _split_module
    """

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.modules_list = modules

    def forward(self, *inputs: Tuple[Tensor]) -> Tensor:  # type: ignore
        input = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            input = module(input)
        return input


def RemoteSequential(rref_list: List[rpc.RRef]) -> MultiInputSequential:
    return MultiInputSequential(*(r.local_value() for r in rref_list))


NodeDataConsumer = DataConsumer["Node"]


@dataclass
class DataSource:
    # If producer is None, we are referring to the model input
    producer: Optional["Node"]
    # indicating which output of the producer module, or which input of the model if producer is None.
    output_idx: int


class Node:
    def __init__(self, module: RemoteModule):
        self.module = module
        self.num_outputs: Optional[int] = None
        self.inputs: List[DataSource] = []
        # To be compiled by _compile method
        self.output_consumers: List[NodeDataConsumer] = []


class PipelineModulesGraph(nn.Module):
    """A collection of remote modules (of type RemoteModule) with connections showing how inputs
    to the model or outputs of individual modules are use as inputs of subsequent modules.
    The graph has a number of helper functions that add new modules to the graph and define inputs
    to these module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nodes: List[Node] = []

    def _find_node(self, module: RemoteModule) -> Node:
        for n in self.nodes:
            if n.module is module:
                return n
        raise ValueError

    def _find_or_add(self, module: RemoteModule) -> Node:
        try:
            return self._find_node(module)
        except ValueError:
            new_node = Node(module)
            self.nodes.append(new_node)
            return new_node

    def _set_inputs(self, module: RemoteModule, inputs: List[DataSource]) -> None:
        self._find_or_add(module).inputs = inputs

    def add_sequence(self, modules: List[RemoteModule], first_input: Optional[RemoteModule] = None) -> None:
        """Adds a list of modules to the graph, to be run sequentially.
        The connection between these modules is as follows: the first output of each of these modules
        (except the last one) is used as the first input of its next module in this sequence.
        The user may also specify the input to the first module in this sequence with argument 'first_input'.
        In this case the module 'first_input' must have been added to the graph previously.
        """
        old_modules_len = len(self.nodes)
        new_modules_len = len(modules)
        self.nodes.extend(Node(mod) for mod in modules)
        # update inputs array
        if first_input is not None:
            self.nodes[old_modules_len].inputs = [DataSource(self._find_node(first_input), 0)]
        for i in range(old_modules_len + 1, old_modules_len + new_modules_len):
            self.nodes[i].inputs = [DataSource(self.nodes[i - 1], 0)]

    def set_model_input(self, module: RemoteModule, ind: int = 0) -> None:
        """Declares the input to a module as the input to the model. In case the model has multiple
        inputs, the argument 'ind' indicates the index of the model input that is fed to the module.
        """
        self._set_inputs(module, [DataSource(None, ind)])

    def add_multi_input_layer(self, module: RemoteModule, inputs: List[RemoteModule]) -> None:
        """Adds a module with multiple inputs to the graph. The modules that provide inputs to this module
        must have been added previously to the graph and are listed with argument inputs.
        """
        self._set_inputs(module, [DataSource(self._find_node(m), 0) for m in inputs])

    def fan_out(self, module: RemoteModule, outputs: List[RemoteModule]) -> None:
        """Feeds outputs of a previously added module to modules specified by argument 'outputs' (so
        'module' should have at least 'len(outputs)' outputs.
        Modules in the list 'outputs' are added to the graph if they have not been added previously.
        """
        node = self._find_node(module)
        node.num_outputs = len(outputs)
        for i, m in enumerate(outputs):
            self._set_inputs(m, [DataSource(node, i)])

    def _compile(self) -> None:
        """Precomputes self.model_input_consumers and self.output_consumers for internal use by the pipleine
        class. These two lists show consumers of inputs to the model, and outputs of each module of
        the graph. Each consumer is a pair (i, j) which stands for the j'th input to the i'th module
        in the graph.
        """

        # TODO: We need to make sure following conditions hold before preparing the graph for the pipeline:
        #   * the graph has a least one module, and is connected.
        #   * num_inputs and num_outputs for modules matche list of connections defined in the graph.
        #   * all inputs to a module should come from model input, or modules with smaller index in
        #     the graph. This condition is used in implementaion of DistributedPipeline.forward. Even
        #     if we relax this condition, still need to make sure the graph is acyclic.

        m = len(self.nodes)
        self.model_input_consumers = []
        for node in self.nodes:
            for input_index, input_item in enumerate(node.inputs):
                data_consumer = NodeDataConsumer(node, input_index, input_item.output_idx)
                if input_item.producer is not None:
                    input_item.producer.output_consumers.append(data_consumer)
                else:
                    self.model_input_consumers.append(data_consumer)

    def _trace_modules(self, node: Node) -> List[Node]:
        """Compiles a list of modules (starting from module number module_idx), where each module in the list
        gets the output of previous module in the list as its input. So every module in the list, except the
        first one should have only one input, and similarly, every module in the list, except the last one
        should have only one output.
        """
        partition = []
        current_node = node
        while True:
            partition.append(current_node)
            # If we reached a module with multiple outputs or with multiple consumers for its output,
            # stop adding more modules to the partition.
            if len(current_node.output_consumers) != 1:
                break
            if current_node.num_outputs is not None:
                break
            # Next module to add is the only consumer of the ouput of the current module
            next_node = current_node.output_consumers[0].consumer
            # If the next module has multiple inputs, do not add it to the current partition and stop.
            if next_node.inputs != [DataSource(current_node, 0)]:
                break
            # If the next module is on a different deivce or worker, stop
            if next_node.module.on != current_node.module.on:
                break
            if next_node.module.device != current_node.module.device:
                break
            current_node = next_node

        return partition

    def partition_graph(self) -> List[Tuple[List[Node], rpc.RRef]]:
        """Splits the graph into pipeline partitions and for each parition returns a tuple (indices, module_rref),
        where indices is indices of modules of the partition in the graph, and module_rref is an RRef to an nn.Module:
        Each partition is a list of modules on the same device that are executed sequentially (output of each module is
        the input to the next module).
        If there is only one module in the partition, module_rref is reference to that module; otherwise those modules
        are wrapped by a MultiInputSequential and module_rref referes to that.
        """
        self._compile()
        modules_used: Set[Node] = set()
        partitions = []
        for node in self.nodes:
            if node in modules_used:
                continue
            partition = self._trace_modules(node)
            assert not modules_used.intersection(partition)
            modules_used.update(partition)

            if len(partition) == 1:
                remote_module = partition[0].module.get_module_rref()
            else:
                remote_module = rpc.remote(
                    partition[0].module.on, RemoteSequential, args=([p.module.get_module_rref() for p in partition],),
                )
            partitions.append((partition, remote_module))

        return partitions
