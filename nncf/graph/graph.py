"""
 Copyright (c) 2019-2020 Intel Corporation
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

import os
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import networkx as nx
import networkx.algorithms.isomorphism as iso
from networkx.drawing.nx_agraph import to_agraph

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.module_attributes import BaseModuleAttributes
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.graph.graph import MODEL_INPUT_OP_NAME
from nncf.common.graph.graph import MODEL_OUTPUT_OP_NAME
from nncf.graph.graph_matching import Expression
from nncf.graph.graph_matching import NodeExpression
from nncf.graph.graph_matching import get_edge_boundaries
from nncf.graph.graph_matching import search_all


# pylint: disable=too-many-public-methods


class InputAgnosticOperationExecutionContext:
    def __init__(self, operator_name: str, scope_in_model: 'Scope', call_order: int):
        self.operator_name = operator_name
        self.scope_in_model = scope_in_model
        self.call_order = call_order

    def __eq__(self, other: 'InputAgnosticOperationExecutionContext'):
        return (self.operator_name == other.operator_name) and \
               (self.scope_in_model == other.scope_in_model) and \
               (self.call_order == other.call_order)

    def __str__(self):
        return str(self.scope_in_model) + '/' + \
               self.operator_name + "_" + str(self.call_order)

    def __hash__(self):
        return hash((self.operator_name, self.scope_in_model, self.call_order))

    @staticmethod
    def from_str(s: str):
        scope_and_op, _, call_order_str = s.rpartition('_')
        scope_str, _, op_name = scope_and_op.rpartition('/')

        from nncf.dynamic_graph.context import Scope
        return InputAgnosticOperationExecutionContext(op_name,
                                                      Scope.from_str(scope_str),
                                                      int(call_order_str))


ModuleAttributes = TypeVar('ModuleAttributes', bound=BaseModuleAttributes)


class PTNNCFNode(NNCFNode):
    def __init__(self, node_id: int,
                 ia_op_exec_context: InputAgnosticOperationExecutionContext,
                 data: dict = None):
        super().__init__(node_id, data)
        self.ia_op_exec_context = ia_op_exec_context

    @property
    def node_type(self):
        if self.ia_op_exec_context:
            return self.ia_op_exec_context.operator_name
        return None

    def __str__(self):
        return str(self.node_id) + ' ' + str(self.ia_op_exec_context)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, PTNNCFNode) \
               and super().__eq__(other) \
               and self.ia_op_exec_context == other.ia_op_exec_context


class NNCFGraphEdge:
    def __init__(self, from_node: PTNNCFNode, to_node: PTNNCFNode, tensor_shape: Tuple):
        self.from_node = from_node
        self.to_node = to_node
        self.tensor_shape = tensor_shape

    def __str__(self):
        return str(self.from_node) + " -> " + str(self.tensor_shape) + " -> " + str(self.to_node)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node \
               and self.tensor_shape == other.tensor_shape


class NNCFGraphPatternIO:
    def __init__(self, input_edges: List[NNCFGraphEdge], output_edges: List[NNCFGraphEdge],
                 input_nodes: List[PTNNCFNode],
                 output_nodes: List[PTNNCFNode],
                 ):
        self.input_edges = input_edges
        self.output_edges = output_edges
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes


class PTNNCFGraph(NNCFGraph):

    IA_OP_EXEC_CONTEXT_NODE_ATTR = 'ia_op_exec_context'
    ACTIVATION_SHAPE_EDGE_ATTR = 'activation_shape'
    IN_PORT_NAME_EDGE_ATTR = 'in_port'

    def __init__(self):
        super().__init__()
        self._input_nncf_nodes = {}  # type: Dict[int, PTNNCFNode]
        self._output_nncf_nodes = {}  # type: Dict[int, PTNNCFNode]

    def __eq__(self, other: 'PTNNCFGraph'):
        nm = iso.categorical_node_match([PTNNCFGraph.ID_NODE_ATTR,
                                         PTNNCFGraph.KEY_NODE_ATTR,
                                         PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR,
                                         PTNNCFGraph.MODULE_ATTRIBUTES], [None, None, None])
        em = iso.categorical_edge_match([PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR,
                                         PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR], [None, None])
        return nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)

    def add_nncf_node(self, nncf_node: PTNNCFNode):
        node_id = nncf_node.node_id
        if node_id in self._node_id_to_key_dict:
            raise ValueError(f"NNCF node with id {node_id} is already in the NNCFGraph")

        name_parts = (str(nncf_node.ia_op_exec_context.scope_in_model),
                      nncf_node.ia_op_exec_context.operator_name)
        node_key = '{idx} {uri}'.format(uri='/'.join(name_parts), idx=node_id)

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            PTNNCFGraph.ID_NODE_ATTR: node_id,
            PTNNCFGraph.KEY_NODE_ATTR: node_key,
            PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR: nncf_node.ia_op_exec_context
        }
        if nncf_node.module_attributes is not None:
            attrs[NNCFGraph.MODULE_ATTRIBUTES] = nncf_node.module_attributes
        self._nx_graph.add_node(node_key, **attrs)

        if nncf_node.node_type == MODEL_INPUT_OP_NAME:
            self._input_nncf_nodes[node_id] = deepcopy(nncf_node)

        if nncf_node.node_type == MODEL_OUTPUT_OP_NAME:
            self._output_nncf_nodes[node_id] = deepcopy(nncf_node)

    def add_edge_between_nncf_nodes(self, from_node_id: int, to_node_id: int,
                                    tensor_shape: List[int],
                                    input_port_id: int):
        from_node_key = self._node_id_to_key_dict[from_node_id]
        to_node_key = self._node_id_to_key_dict[to_node_id]

        err_reason = None

        if from_node_key not in self._nx_graph.nodes:
            err_reason = f"node {from_node_key} not in NNCFGraph"
        if to_node_key not in self._nx_graph.nodes:
            err_reason = f"node {from_node_key} not in NNCFGraph"
        if from_node_id in self._output_nncf_nodes:
            err_reason = "cannot add edges *from* output nodes"
        if to_node_id in self._input_nncf_nodes:
            err_reason = "cannot add edges *to* input nodes"

        if err_reason is not None:
            raise ValueError(f"Cannot add edge from {from_node_key} to {to_node_key} - {err_reason}!")

        attrs = {
            PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: tensor_shape,
            PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: input_port_id
        }
        self._nx_graph.add_edge(from_node_key, to_node_key, **attrs)

    def get_input_nodes(self) -> List[PTNNCFNode]:
        return list(self._input_nncf_nodes.values())

    def get_output_nodes(self) -> List[PTNNCFNode]:
        return list(self._output_nncf_nodes.values())

    def get_nncf_node_by_id(self, node_id: int) -> PTNNCFNode:
        nx_node = self.get_nx_node_by_key(self.get_node_key_by_id(node_id))
        nncf_node = self._nx_node_to_nncf_node(nx_node)
        return nncf_node

    def get_nx_node_by_key(self, key: str) -> dict:
        return self._nx_graph.nodes[key]

    def get_node_key_by_iap_context(self, iap_ctx: InputAgnosticOperationExecutionContext) -> str:
        for node_key, node in self._nx_graph.nodes.items():
            if node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR] == iap_ctx:
                return node_key
        raise AttributeError('Failed to get node by context={}'.format(str(iap_ctx)))

    def get_successors(self, node_name: str):
        return self._nx_graph.successors(node_name)

    def get_successor_nncf_nodes(self, node_id: int) -> List[NNCFNode]:
        key = self.get_node_key_by_id(node_id)
        succs = list(self._nx_graph.successors(key))
        nncf_nodes = [self._nx_node_to_nncf_node(self._nx_graph.nodes[nx_node_key]) for nx_node_key in succs]
        return nncf_nodes

    def get_matching_nncf_graph_pattern_io_list(self, expression: Expression) -> List[NNCFGraphPatternIO]:
        matched_node_key_sequences = search_all(self._nx_graph, expression)
        pattern_ios = [self._get_nncf_graph_pattern_io_list(match) for match in matched_node_key_sequences]
        return pattern_ios

    def dump_graph(self, path):
        nx.drawing.nx_pydot.write_dot(self._get_graph_for_structure_analysis(), path)

    def visualize_graph(self, path):
        out_graph = self._get_graph_for_visualization()
        nx.drawing.nx_pydot.write_dot(out_graph, path)
        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            nncf_logger.warning("Graphviz is not installed - only the .dot model visualization format will be used. "
                                "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                                "PNG rendering.")

    def is_output_node(self, node: PTNNCFNode) -> bool:
        return not list(self._nx_graph.successors(self._node_id_to_key_dict[node.node_id]))

    def get_nx_graph_copy(self) -> nx.DiGraph:
        return deepcopy(self._nx_graph)

    # pylint:disable=protected-access
    def get_nx_edge(self, node_u: PTNNCFNode, node_v: PTNNCFNode):
        nx_node_u = self._nx_graph._node[self._node_id_to_key_dict[node_u.node_id]]
        nx_node_v = self._nx_graph._node[self._node_id_to_key_dict[node_v.node_id]]
        return self._nx_graph.edges[nx_node_u['key'], nx_node_v['key']]

    def get_inputs_count(self, node: PTNNCFNode) -> int:
        return self._nx_graph.in_degree()[self._node_id_to_key_dict[node.node_id]]

    def get_nodes_count(self):
        return self._nx_graph.number_of_nodes()

    @staticmethod
    def node_type_fn(node: dict) -> str:
        return node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].operator_name

    def get_output_shapes_for_ia_op_exec_context(self,
                                                 ia_op_exec_context: InputAgnosticOperationExecutionContext)\
                                                 -> List[Tuple]:
        node_key = self.get_node_key_by_iap_context(ia_op_exec_context)
        succs = list(self._nx_graph.successors(node_key))
        edge_list = [self._nx_graph.edges[node_key, to_node_key] for to_node_key in succs]
        return [edge[PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR] for edge in edge_list]

    def get_input_shapes_for_ia_op_exec_context(self,
                                                ia_op_exec_context: InputAgnosticOperationExecutionContext) \
            -> Dict[int, Tuple]:
        node_key = self.get_node_key_by_iap_context(ia_op_exec_context)
        in_edges = list(self._nx_graph.in_edges(node_key))
        retval = {}
        for in_edge in in_edges:
            edge_attr_dict = self._nx_graph.edges[in_edge]
            port_id = edge_attr_dict[PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR]
            assert port_id not in retval
            retval[port_id] = edge_attr_dict[PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
        return retval

    def _get_graph_for_structure_analysis(self, extended=False) -> nx.DiGraph:
        """The graph to dump has certain node attributes omitted, compared to the graph stored
         inside NNCFGraph."""
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            ia_op_exec_context = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]
            scope_str = str(ia_op_exec_context.scope_in_model)
            attrs_node = {
                'type': ia_op_exec_context.operator_name,
                'id': node[PTNNCFGraph.ID_NODE_ATTR],
                'scope': scope_str
            }
            if 'color' in node:
                attrs_node['color'] = node['color']
            if 'label' in node:
                attrs_node['label'] = node['label']
            if 'style' in node:
                attrs_node['style'] = node['style']

            out_graph.add_node(node_name, **attrs_node)
        if extended:
            for u, v in self._nx_graph.edges:
                out_graph.add_edge(u, v, label=self._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
        else:
            for u, v in self._nx_graph.edges:
                out_graph.add_edge(u, v)

        return out_graph

    def _get_graph_for_visualization(self) -> nx.DiGraph:
        """A user-friendly graph .dot file, making it easier to debug the network and setup
        ignored/target scopes."""
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            ia_op_exec_context = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]

            attrs_node = {}
            attrs_node['label'] = str(node[PTNNCFGraph.ID_NODE_ATTR]) + ' ' + str(ia_op_exec_context)

            out_graph.add_node(node_name, **attrs_node)

        for u, v in self._nx_graph.edges:
            out_graph.add_edge(u, v, label=self._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        return out_graph

    def _get_topologically_last_nodes(self, matches: List[List[str]]) -> List[str]:
        topological_order = {node: k for k, node in enumerate(nx.topological_sort(self._nx_graph))}
        insertion_points = {max(match, key=topological_order.__getitem__) for match in matches}
        for match in matches:
            for node in match:
                if len(list(self._nx_graph.successors(node))) > 1:
                    insertion_points.add(node)

        return list(insertion_points)

    def _get_nncf_graph_pattern_input_output(self, match: List[str]) -> NNCFGraphPatternIO:
        out_edge_boundary = list(nx.edge_boundary(self._nx_graph, match, data=True))
        complement = list(filter(lambda x: x not in match, self._nx_graph.nodes.keys()))
        in_edge_boundary = list(nx.edge_boundary(self._nx_graph, complement, data=True))
        boundary = in_edge_boundary + out_edge_boundary
        input_nncf_edges = []
        output_nncf_edges = []
        input_nncf_nodes = []
        output_nncf_nodes = []
        for key in match:
            # Currently we treat the nodes without incoming edges as "input" and the nodes without
            # outcoming edges as "output".
            # A proper way to find the input nodes would be to mark the tensors arriving at NNCFNetwork's
            # "forward" as input, then drop the marking once the first operation with an input tensor
            # has been done; the node corresponding to this operation would be "input" by definition.
            # Same with output nodes - should check the model output for TracedTensors and mark the
            # nodes from which such tensors originated as "output".
            # TODO: implement the functionality above.
            if not list(self._nx_graph.successors(key)):
                output_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))
            if not list(self._nx_graph.predecessors(key)):
                input_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))

        for nx_edge in boundary:
            from_node_key = nx_edge[0]
            to_node_key = nx_edge[1]
            data = nx_edge[2]
            nncf_edge = NNCFGraphEdge(self._nx_node_to_nncf_node(self._nx_graph.nodes[from_node_key]),
                                      self._nx_node_to_nncf_node(self._nx_graph.nodes[to_node_key]),
                                      data[PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            if from_node_key in match:
                output_nncf_edges.append(nncf_edge)
            elif to_node_key in match:
                input_nncf_edges.append(nncf_edge)
            else:
                raise RuntimeError("Invalid graph expression supplied!")

        return NNCFGraphPatternIO(input_nncf_edges, output_nncf_edges,
                                  input_nncf_nodes, output_nncf_nodes)

    def _get_nncf_graph_pattern_io_list(self, match: List[str]) -> NNCFGraphPatternIO:
        in_edge_boundary, out_edge_boundary = get_edge_boundaries(match, self._nx_graph)
        boundary = in_edge_boundary + out_edge_boundary
        input_nncf_edges = []
        output_nncf_edges = []
        input_nncf_nodes = []
        output_nncf_nodes = []
        for key in match:
            # Currently we treat the nodes without incoming edges as "input" and the nodes without
            # outcoming edges as "output".
            # A proper way to find the input nodes would be to mark the tensors arriving at NNCFNetwork's
            # "forward" as input, then drop the marking once the first operation with an input tensor
            # has been done; the node corresponding to this operation would be "input" by definition.
            # Same with output nodes - should check the model output for TracedTensors and mark the
            # nodes from which such tensors originated as "output".
            # TODO: implement the functionality above.
            if not list(self._nx_graph.successors(key)):
                output_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))
            if not list(self._nx_graph.predecessors(key)):
                input_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))

        for nx_edge in boundary:
            from_node_key = nx_edge[0]
            to_node_key = nx_edge[1]
            data = nx_edge[2]
            nncf_edge = NNCFGraphEdge(self._nx_node_to_nncf_node(self._nx_graph.nodes[from_node_key]),
                                      self._nx_node_to_nncf_node(self._nx_graph.nodes[to_node_key]),
                                      data[PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            if from_node_key in match:
                output_nncf_edges.append(nncf_edge)
            elif to_node_key in match:
                input_nncf_edges.append(nncf_edge)
            else:
                raise RuntimeError("Invalid graph expression supplied!")

        return NNCFGraphPatternIO(input_nncf_edges, output_nncf_edges,
                                  input_nncf_nodes, output_nncf_nodes)

    @staticmethod
    def _nx_node_to_nncf_node(nx_node) -> 'PTNNCFNode':
        return PTNNCFNode(nx_node[PTNNCFGraph.ID_NODE_ATTR],
                          nx_node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR],
                          nx_node)

    def find_node_in_nx_graph_by_scope(self, scope: 'Scope') -> Optional[PTNNCFNode]:
        """
        Looking for node with scope == scope in networkx graph.
        :param self: graphs to work on
        :param scope: Scope to find in graph
        :return: node from networkx graph for graph (or None if such scope is not found)
        """
        nodes = self._nx_graph.nodes
        for node_key in nodes:
            if nodes[node_key][PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model == scope:
                return self._nx_node_to_nncf_node(nodes[node_key])
        return None

    def find_node_in_nx_graph_by_input_agnostic(
        self, input_agnostic: InputAgnosticOperationExecutionContext
    ) -> Optional[dict]:
        """
        Looking for node with input_agnostic == input_agnostic in networkx graph.
        :param self: graphs to work on
        :param input_agnostic: Input agnostic to find in graph
        :return: node from networkx graph for graph (or None if such scope is not found)
        """
        nodes = self._nx_graph.nodes
        for node_key in nodes:
            if nodes[node_key][PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR] == input_agnostic:
                return nodes[node_key]
        return None

    def get_op_nodes_in_scope(self, scope: 'Scope') -> List:
        matching_graph_op_nodes = []
        for _, node in self._nx_graph.nodes.items():
            op_scope = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model
            if op_scope in scope:
                matching_graph_op_nodes.append(node)
        return matching_graph_op_nodes


class NNCFNodeExpression(NodeExpression):
    def __init__(self, node_type: str = None, filter_fn=None):
        super().__init__(node_type, filter_fn, node_type_fn=PTNNCFGraph.node_type_fn)


def get_module_identifier(node: PTNNCFNode):
    return str(node.ia_op_exec_context.scope_in_model)
