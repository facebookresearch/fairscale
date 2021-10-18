# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

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
        self.modules_list = nn.ModuleList(modules)

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

    # DataSourceSpec lists choices the user has for specifying the source of each input to a module:
    # -- If the input is one of model inputs, it is specified by a simple integer, which is the index of that input
    # -- If the input comes from a module with a simple output, it is specified by that module
    # -- If the input comes from a module with multiple outputs (a tuple), it is specified by that module and the
    #    index of the output
    DataSourceSpec = Union[int, RemoteModule, Tuple[RemoteModule, int]]

    def _data_source_spec_to_data_source(self, spec: DataSourceSpec) -> DataSource:
        if isinstance(spec, int):
            return DataSource(None, spec)
        if isinstance(spec, RemoteModule):
            return DataSource(self._find_node(spec), 0)
        return DataSource(self._find_node(spec[0]), spec[1])

    def add_layer(self, module: RemoteModule, inputs: List[DataSourceSpec], num_outputs: Optional[int] = None) -> None:
        """Adds a module with specified inputs to the graph. The modules that provide inputs to this module must have
        been added previously to the graph and are listed with argument inputs. If the module output is a tuple,
        num_outputs specifies the number of elements in the tuple.
        """
        node = Node(module)
        node.inputs = [self._data_source_spec_to_data_source(spec) for spec in inputs]
        node.num_outputs = num_outputs
        self.nodes.append(node)

    def add_sequence(
        self,
        modules: List[RemoteModule],
        first_module_inputs: List[DataSourceSpec],
        last_module_num_outputs: Optional[int] = None,
    ) -> None:
        """Adds a list of modules to the graph, to be run sequentially.
        The connection between these modules is as follows: the output of each of these modules
        (except the last one) is used as the input of its next module in this sequence.
        So all modules (except the last one) must have simple output, and also all of them (except the first one)
        should have a single input.
        The user also specifies the input to the first module in this sequence with argument 'first_module_inputs'.
        In case the last module output is a tuple, 'last_module_num_outputs' specifies the number of elements
        in the tuple.
        """
        next_input = first_module_inputs
        for i, module in enumerate(modules):
            self.add_layer(module, next_input, last_module_num_outputs if i == len(modules) - 1 else None)
            next_input = [module]

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
