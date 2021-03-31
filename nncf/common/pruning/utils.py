"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import math
from functools import partial
from typing import List, Tuple, Optional

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.module_attributes import ConvolutionModuleAttributes
from nncf.common.utils.registry import Registry


def is_grouped_conv(node: NNCFNode) -> bool:
    return isinstance(node.module_attributes, ConvolutionModuleAttributes) \
           and node.module_attributes.groups != 1


def get_sources_of_node(nncf_node: NNCFNode, graph: NNCFGraph, sources_types: List[str]) -> List[NNCFNode]:
    """
    Source is a node of source such that there is path from this node to `nncf_node` and on this path
    no node has one of `sources_types` type.

    :param sources_types: List of sources types.
    :param nncf_node: NNCFNode to get sources.
    :param graph: NNCF graph to work with.
    :return: List of all sources nodes.
    """
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in sources_types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type in sources_types:
        nncf_nodes = graph.get_previous_nodes(nncf_node)

    source_nodes = []
    for node in nncf_nodes:
        source_nodes.extend(graph.traverse_graph(node, partial_traverse_function, False))
    return source_nodes


def find_next_nodes_not_of_types(graph: NNCFGraph, nncf_node: NNCFNode, types: List[str]) -> List[NNCFNode]:
    """
    Traverse nodes in the graph from nncf node to find first nodes that aren't of type from types list.
    First nodes with some condition mean nodes:
        - for which this condition is true
        - reachable from `nncf_node` such that on the path from `nncf_node` to
          this nodes there are no other nodes with fulfilled condition

    :param graph: Graph to work with.
    :param nncf_node: NNCFNode to start search.
    :param types: List of types.
    :return: List of next nodes for nncf_node of type not from types list.
    """
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x not in types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type not in types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes


def get_next_nodes_of_types(graph: NNCFGraph, nncf_node: NNCFNode, types: List[str]) -> List[NNCFNode]:
    """
    Looking for nodes with type from types list from `nncf_node` such that there is path
    from `nncf_node` to this node and on this path no node has one of types type.

    :param graph: Graph to work with.
    :param nncf_node: NNCFNode to start search.
    :param types: List of types to find.
    :return: List of next nodes of nncf_node with type from types list.
    """
    sources_types = types
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in sources_types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type in sources_types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes


def get_rounded_pruned_element_number(total: int, sparsity_rate: float, multiple_of: int = 8) -> int:
    """
    Calculates number of sparsified elements (approximately sparsity rate) from total such as
    number of remaining items will be multiple of some value.
    Always rounds number of remaining elements up.

    :param total: Total elements number.
    :param sparsity_rate: Prorortion of zero elements in total.
    :param multiple_of: Number of remaining elements must be a multiple of `multiple_of`.
    :return: Number of elements to be zeroed.
    """
    remaining_elems = math.ceil((total - total * sparsity_rate) / multiple_of) * multiple_of
    return max(total - remaining_elems, 0)


def traverse_function(node: NNCFNode, output: List[NNCFNode], type_check_fn, visited) \
        -> Tuple[bool, List[NNCFNode]]:
    if visited[node.node_id]:
        return True, output
    visited[node.node_id] = True

    if not type_check_fn(node.node_type):
        return False, output

    output.append(node)
    return True, output


def get_first_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for first node in graph with type in `op_types`.
    First == layer with type in `op_types`, that there is a path from the input such that there are no other
    operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all first nodes with type in `op_types`.
    """
    graph_roots = graph.get_input_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function,
                                        type_check_fn=lambda x: x in op_types,
                                        visited=visited)

    first_nodes_of_type = []
    for root in graph_roots:
        first_nodes_of_type.extend(graph.traverse_graph(root, partial_traverse_function))
    return first_nodes_of_type


def get_last_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for last node in graph with type in `op_types`.
    Last == layer with type in `op_types`, that there is a path from this layer to the model output
    such that there are no other operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all last pruned nodes.
    """
    graph_outputs = graph.get_output_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function,
                                        type_check_fn=lambda x: x in op_types,
                                        visited=visited)
    last_nodes_of_type = []
    for output in graph_outputs:
        last_nodes_of_type.extend(graph.traverse_graph(output, partial_traverse_function, False))

    return last_nodes_of_type


def get_previous_conv(graph: NNCFGraph, nncf_node: NNCFNode,
                      pruning_types: List[str], stop_propagation_ops: List[str]) -> Optional[NNCFNode]:
    """
    Returns source convolution of the node. If the node has another source type or there is
    more than one source - returns None.

    :return: Source convolution of node. If the node has another source type or there is more
        than one source - returns None.
    """
    sources = get_sources_of_node(nncf_node, graph, pruning_types + stop_propagation_ops)
    if len(sources) == 1 and sources[0].node_type in pruning_types:
        return sources[0]
    return None


class PruningOperationsMetatypeRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._op_name_to_op_class = {}

    def register(self, name=None):
        name_ = name
        super_register = super()._register

        def wrap(obj):
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__

            super_register(obj, cls_name)
            op_names = obj.get_all_op_aliases()
            for name in op_names:
                name = self.get_version_agnostic_name(name)
                if name not in self._op_name_to_op_class:
                    self._op_name_to_op_class[name] = obj
                else:
                    assert self._op_name_to_op_class[name] == obj, \
                        "Inconsistent operator type registry - single patched op name maps to multiple metatypes!"
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str):
        if op_name in self._op_name_to_op_class:
            return self._op_name_to_op_class[op_name]
        return None

    @staticmethod
    def get_version_agnostic_name(name):
        raise NotImplementedError
