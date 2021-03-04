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

from typing import List, Optional, Tuple, Dict
import os

import networkx as nx
import networkx.algorithms.isomorphism as iso

from copy import deepcopy
from networkx.drawing.nx_agraph import to_agraph
from torch import Tensor

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.dynamic_graph.graph_matching import Expression, NodeExpression, search_all, get_edge_boundaries
from nncf.dynamic_graph.trace_tensor import TensorMeta, TracedTensor
from nncf.layers import ITERATION_MODULES
from nncf.common.utils.logger import logger as nncf_logger


# pylint: disable=too-many-public-methods
class TensorMetaComparator:
    def __call__(self, lhs: TensorMeta, rhs: TensorMeta) -> bool:
        raise NotImplementedError


class DefaultTensorMetaComparator(TensorMetaComparator):
    def __call__(self, lhs: TensorMeta, rhs: TensorMeta) -> bool:
        return TensorMeta.default_comparator(lhs, rhs)


class ShapeIgnoringTensorMetaComparator(TensorMetaComparator):
    def __call__(self, lhs: TensorMeta, rhs: TensorMeta) -> bool:
        return lhs.creator_id == rhs.creator_id and lhs.index == rhs.index


class ShapeOnlyTensorMetaComparator(TensorMetaComparator):
    def __call__(self, lhs: TensorMeta, rhs: TensorMeta) -> bool:
        return lhs.shape[1:] == rhs.shape[1:]


class InputsMatcher:
    def __call__(self, node_inputs: List[TensorMeta], real_inputs: List[TensorMeta],
                 tm_comparators: List[TensorMetaComparator]) -> bool:
        raise NotImplementedError


class FirstInputsMatcher(InputsMatcher):
    def __call__(self, node_inputs: List[TensorMeta], real_inputs: List[TensorMeta],
                 tm_comparators: List[TensorMetaComparator]) -> bool:
        if not node_inputs or not real_inputs:
            return False

        if not node_inputs[0] or not real_inputs[0]:
            return False

        for tm_comparator in tm_comparators:
            if not tm_comparator(node_inputs[0], real_inputs[0]):
                return False
        return True


class DefaultInputsMatcher(InputsMatcher):
    def __call__(self, node_inputs: List[TensorMeta], real_inputs: List[TensorMeta],
                 tm_comparators: List[TensorMetaComparator]) -> bool:
        if node_inputs is None and real_inputs:
            return False

        for saved_input, actual_input in zip(node_inputs, real_inputs):
            if saved_input is None and actual_input is None:
                continue
            if (saved_input is None) != (actual_input is None):
                return False
            for tm_comparator in tm_comparators:
                if not tm_comparator(saved_input, actual_input):
                    return False
        return True


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


class OperationExecutionContext:
    """Information that allows to uniquely identify an operation inside the NNCF graph,
    i.e. determine whether an execution of the operator inside the module has already been
    registered as a node in the graph or not (in the latter case a new node would have to
    be created"""

    def __init__(self,
                 operator_name: str,
                 scope_in_model: 'Scope',
                 call_order: int,
                 tensor_metas: List[TensorMeta],
                 tm_comparators: List[TensorMetaComparator] = None,
                 input_matcher: InputsMatcher = None):
        self.input_agnostic = InputAgnosticOperationExecutionContext(operator_name, scope_in_model, call_order)
        # This should be a list with a length equal to the number of inputs.
        # "None" values in this list correspond to non-tensor input elements.
        self.tensor_metas = tensor_metas
        self.tm_comparators = tm_comparators if tm_comparators else [
            DefaultTensorMetaComparator()]
        self.input_matcher = input_matcher if input_matcher else DefaultInputsMatcher()

    def __eq__(self, other: 'OperationExecutionContext'):
        return (self.input_agnostic == other.input_agnostic) and \
               self.input_matcher(self.tensor_metas, other.tensor_metas, self.tm_comparators)

    def __hash__(self):
        return hash((self.operator_name, tuple(self.scope_in_model), self.call_order,
                     tuple(self.tensor_metas)))

    def __str__(self):
        input_info_str = ""
        for meta in self.tensor_metas:
            if meta is None:
                input_info_str += "N;"
            else:
                input_info_str += str(meta) + ";"

        return super().__str__() + '(' + input_info_str + ')'

    @property
    def operator_name(self):
        return self.input_agnostic.operator_name

    @property
    def scope_in_model(self) -> 'Scope':
        return self.input_agnostic.scope_in_model

    @property
    def call_order(self):
        return self.input_agnostic.call_order


class PTNNCFNode(NNCFNode):
    def __init__(self, node_id: int, op_exec_context: OperationExecutionContext, data: dict = None):
        super().__init__(node_id, data)
        self.op_exec_context = op_exec_context

    @property
    def node_type(self):
        if self.op_exec_context:
            return self.op_exec_context.input_agnostic.operator_name
        return None

    def __str__(self):
        return str(self.node_id) + ' ' + str(self.op_exec_context)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, PTNNCFNode) \
               and super().__eq__(other) \
               and self.op_exec_context == other.op_exec_context


class DefaultScopeNodeMatcher:
    def __init__(self, node_id_to_key_dict, nx_graph, nx_node_to_nncf_node):
        self._node_id_to_key_dict = node_id_to_key_dict
        self._nx_graph = nx_graph
        self._nx_node_to_nncf_node = nx_node_to_nncf_node
        self._inputless_nx_nodes = dict()

    def get_node_by_id(self, node_id):
        return self._nx_graph.nodes[self._node_id_to_key_dict[node_id]]

    def _find_nodes_with_matching_context_among_inputless(self, op_exec_context: OperationExecutionContext):
        node_candidates = {}
        for nx_node_key, nx_node in self._inputless_nx_nodes.items():
            if nx_node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR] == op_exec_context:
                node_candidates[nx_node_key] = nx_node
        return node_candidates

    def _find_nodes_with_matching_context_and_inputs(self, op_exec_context: OperationExecutionContext):
        node_candidates = {}
        for info in op_exec_context.tensor_metas:
            if info is None or info.creator_id is None:
                continue
            creator_id = info.creator_id
            for successor_node_key in self._nx_graph.successors(self._node_id_to_key_dict[creator_id]):
                successor_node = self._nx_graph.nodes[successor_node_key]
                if op_exec_context == successor_node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                    node_candidates[successor_node_key] = successor_node
        return node_candidates

    def add_node(self, op_exec_context: OperationExecutionContext, inputs) -> PTNNCFNode:
        node_id = len(self._node_id_to_key_dict)

        name_parts = (str(op_exec_context.scope_in_model), op_exec_context.operator_name)
        node_key = '{idx} {uri}'.format(uri='/'.join(name_parts), idx=node_id)

        nncf_logger.debug("New node added to NNCF graph: {}".format(node_key))

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            PTNNCFGraph.ID_NODE_ATTR: node_id,
            PTNNCFGraph.KEY_NODE_ATTR: node_key,
            PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: op_exec_context,
        }
        self._nx_graph.add_node(node_key, **attrs)

        has_traced_inputs = False
        for i, info in enumerate(op_exec_context.tensor_metas):
            if info is None or info.creator_id is None:
                continue
            parent = self._node_id_to_key_dict[info.creator_id]
            self._nx_graph.add_edge(parent, node_key)
            has_traced_inputs = True
            self._nx_graph.edges[parent, node_key][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR] = info.shape
            self._nx_graph.edges[parent, node_key][PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR] = i

        if not has_traced_inputs:
            self._inputless_nx_nodes[node_key] = self._nx_graph.nodes[node_key]

        return self._nx_node_to_nncf_node(self._nx_graph.nodes[node_key])

    def find_node(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                  tensor_metas: List[TensorMeta],
                  tm_comparators: List[TensorMetaComparator]) -> PTNNCFNode:
        op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                    ia_op_exec_context.scope_in_model,
                                                    ia_op_exec_context.call_order,
                                                    tensor_metas,
                                                    tm_comparators=tm_comparators)
        nncf_node_candidates = []
        node_candidates = self._find_nodes_with_matching_context_and_inputs(op_exec_context)
        if not node_candidates:
            node_candidates = self._find_nodes_with_matching_context_among_inputless(op_exec_context)

        for nx_node in node_candidates.values():
            nncf_node_candidates.append(PTNNCFNode(nx_node[PTNNCFGraph.ID_NODE_ATTR],
                                                   op_exec_context,
                                                   nx_node))
        result = None
        if len(nncf_node_candidates) == 1:
            result = nncf_node_candidates[0]
        if len(nncf_node_candidates) > 1:
            nncf_logger.warning("More than one node matches input")
            result = nncf_node_candidates[0]

        return result


class IterationScopeNodeMatcher(DefaultScopeNodeMatcher):
    def __init__(self, node_id_to_key_dict, nx_graph, nx_node_to_nncf_node):
        super().__init__(node_id_to_key_dict, nx_graph, nx_node_to_nncf_node)
        self._first_iteration_nodes = {}  # type: {str: {str: PTNNCFNode}}

    def save_first_iteration_node(self, inputs, node: PTNNCFNode):
        """
        It finds and saves "starting" points of iteration for further matching with them on next iteration,
        instead of adding new nodes for each iteration. "Starting" points of iteration are nodes
            * that have at least one input node, which is outside of iteration scope
            * or whose all inputs are not TracedTensor
        """
        op_exec_context = node.op_exec_context
        name = node
        iter_scopes = op_exec_context.scope_in_model.get_iteration_scopes()
        if iter_scopes:
            for iter_scope in iter_scopes:
                if iter_scope not in self._first_iteration_nodes:
                    self._first_iteration_nodes[iter_scope] = {}
                first_nodes = self._first_iteration_nodes[iter_scope]
                has_input_outside_iteration = False
                not_traced_count = 0
                for i in inputs:
                    if isinstance(i, Tensor):
                        has_input_outside_iteration = True
                        break
                    if not isinstance(i, TracedTensor):
                        not_traced_count += 1
                        continue
                    creator_id = i.tensor_meta.creator_id
                    creator_node = self.get_node_by_id(creator_id)
                    creator_node_op_exec_ctx = creator_node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                    within_scopes = creator_node_op_exec_ctx.scope_in_model.get_iteration_scopes()
                    if iter_scope not in within_scopes:
                        has_input_outside_iteration = True
                if not_traced_count == len(inputs):
                    has_input_outside_iteration = True
                if has_input_outside_iteration:
                    node_name = str(op_exec_context.input_agnostic)
                    first_nodes[node_name] = node
                    nncf_logger.debug('Found first iteration node: {} in scope: {}'.format(name, iter_scope))

    def add_node(self, op_exec_context: OperationExecutionContext, inputs) -> PTNNCFNode:
        node = super().add_node(op_exec_context, inputs)
        self.save_first_iteration_node(inputs, node)
        return node

    def find_node(self,
                  ia_op_exec_context: InputAgnosticOperationExecutionContext,
                  tensor_metas: List[TensorMeta],
                  tm_comparators: List[TensorMetaComparator]) -> PTNNCFNode:
        nncf_node_candidates = []
        iter_scopes = ia_op_exec_context.scope_in_model.get_iteration_scopes()
        # compare meta information about first input nodes during the matching. During the iteration some nodes may
        # change number of inputs, e.g. on concat of hidden outputs
        input_matcher = FirstInputsMatcher()
        op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                    ia_op_exec_context.scope_in_model,
                                                    ia_op_exec_context.call_order,
                                                    tensor_metas,
                                                    input_matcher=input_matcher,
                                                    tm_comparators=tm_comparators)
        node_candidates = self._find_nodes_with_matching_context_and_inputs(op_exec_context)
        if not node_candidates:
            op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                        ia_op_exec_context.scope_in_model,
                                                        ia_op_exec_context.call_order,
                                                        tensor_metas,
                                                        tm_comparators=tm_comparators)
            node_candidates = self._find_nodes_with_matching_context_among_inputless(op_exec_context)
            if not node_candidates and iter_scopes:
                # ignore information about node creator and index of input
                comparators = tm_comparators + [ShapeOnlyTensorMetaComparator()]
                op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                            ia_op_exec_context.scope_in_model,
                                                            ia_op_exec_context.call_order,
                                                            tensor_metas,
                                                            tm_comparators=comparators)
                # match with starting points of iteration
                iter_nodes = self._match_first_iteration_nodes(op_exec_context, iter_scopes)
                for node in iter_nodes.items():
                    nncf_node_candidates.append(node[1])

        for nx_node in node_candidates.values():
            nncf_node_candidates.append(PTNNCFNode(nx_node[PTNNCFGraph.ID_NODE_ATTR],
                                                   op_exec_context,
                                                   nx_node))

        result = None
        if len(nncf_node_candidates) == 1:
            result = nncf_node_candidates[0]
        if len(nncf_node_candidates) > 1:
            nncf_logger.warning("More than one node matches input")
            result = nncf_node_candidates[0]

        return result

    def _match_first_iteration_nodes(self, op_exec_context: OperationExecutionContext, iter_scopes):
        node_candidates = {}
        for iter_scope in iter_scopes:
            if iter_scope in self._first_iteration_nodes:
                for name, node in self._first_iteration_nodes[iter_scope].items():
                    if op_exec_context == node.op_exec_context:
                        node_candidates[name] = node
                        break
                if node_candidates:
                    break
        return node_candidates


class NodeManager:
    def __init__(self, node_id_to_key_dict, nx_graph, nx_node_to_nncf_node):
        self.base_matcher = DefaultScopeNodeMatcher(node_id_to_key_dict, nx_graph, nx_node_to_nncf_node)
        self.iteration_matcher = IterationScopeNodeMatcher(node_id_to_key_dict, nx_graph, nx_node_to_nncf_node)

    # TODO: optimize by matching exact module type
    @staticmethod
    def _within_iteration(scope: 'Scope'):
        scope_name = str(scope)
        for iter_scope in ITERATION_MODULES.registry_dict:
            if iter_scope in scope_name:
                return True
        return False

    def choose_matcher(self, ia_op_exec_context: InputAgnosticOperationExecutionContext) -> DefaultScopeNodeMatcher:
        if self._within_iteration(ia_op_exec_context.scope_in_model):
            return self.iteration_matcher
        return self.base_matcher

    @staticmethod
    def choose_tm_comparators(ia_op_exec_context: InputAgnosticOperationExecutionContext,
                              input_comparators_per_scope:
                              List[Tuple[TensorMetaComparator, List[str]]]) -> List[TensorMetaComparator]:
        result = []
        for pairs in input_comparators_per_scope:
            comparator, scopes = pairs
            for scope in scopes:
                if scope in str(ia_op_exec_context):
                    result.append(comparator)
        return result

    def find_node(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                  tensor_metas: List[TensorMeta],
                  input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]]) -> PTNNCFNode:
        matcher = self.choose_matcher(ia_op_exec_context)
        comparators = self.choose_tm_comparators(ia_op_exec_context, input_comparators_per_scope)
        return matcher.find_node(ia_op_exec_context, tensor_metas, comparators)

    def add_node(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                 tensor_metas: List[TensorMeta],
                 tm_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
                 inputs) -> PTNNCFNode:
        matcher = self.choose_matcher(ia_op_exec_context)
        tm_comparators = self.choose_tm_comparators(ia_op_exec_context, tm_comparators_per_scope)
        op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                    ia_op_exec_context.scope_in_model,
                                                    ia_op_exec_context.call_order,
                                                    tensor_metas,
                                                    tm_comparators=tm_comparators)

        return matcher.add_node(op_exec_context, inputs)


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

    OP_EXEC_CONTEXT_NODE_ATTR = 'op_exec_context'
    ACTIVATION_SHAPE_EDGE_ATTR = 'activation_shape'
    IN_PORT_NAME_EDGE_ATTR = 'in_port'

    def __init__(self):
        super().__init__()
        self.match_manager = NodeManager(self._node_id_to_key_dict, self._nx_graph, self._nx_node_to_nncf_node)
        self._input_nncf_nodes = []

    def __eq__(self, other: 'PTNNCFGraph'):
        nm = iso.categorical_node_match([PTNNCFGraph.ID_NODE_ATTR,
                                         PTNNCFGraph.KEY_NODE_ATTR,
                                         PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR,
                                         PTNNCFGraph.MODULE_ATTRIBUTES], [None, None, None])
        em = iso.categorical_edge_match([PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR,
                                         PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR], [None, None])
        return nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)

    def find_node(self,
                  ia_op_exec_context: InputAgnosticOperationExecutionContext,
                  tensor_metas: List[TensorMeta],
                  input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]]) -> PTNNCFNode:
        return self.match_manager.find_node(ia_op_exec_context, tensor_metas, input_comparators_per_scope)

    def add_node(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                 tensor_metas: List[TensorMeta],
                 input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
                 inputs) -> PTNNCFNode:
        node = self.match_manager.add_node(ia_op_exec_context, tensor_metas, input_comparators_per_scope, inputs)

        from nncf.dynamic_graph.input_wrapping import MODEL_INPUT_OP_NAME
        if node.op_exec_context.operator_name == MODEL_INPUT_OP_NAME:  # TODO: refactorable model input node name
            self._input_nncf_nodes.append(node)
        return node

    def get_input_nodes(self) -> List[PTNNCFNode]:
        return self._input_nncf_nodes

    def get_nncf_node_by_id(self, node_id: int) -> PTNNCFNode:
        nx_node = self.get_nx_node_by_key(self.get_node_key_by_id(node_id))
        nncf_node = self._nx_node_to_nncf_node(nx_node)
        return nncf_node

    def get_node_key_by_iap_context(self, iap_ctx: InputAgnosticOperationExecutionContext) -> str:
        for node_key, node in self._nx_graph.nodes.items():
            if node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic == iap_ctx:
                return node_key
        raise AttributeError('Failed to get node by context={}'.format(str(iap_ctx)))

    def get_successors(self, node_name: str):
        return self._nx_graph.successors(node_name)

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
        return node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name

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
            op_exec_context = node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
            scope_str = str(op_exec_context.scope_in_model)
            attrs_node = {
                'type': op_exec_context.operator_name,
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
            op_exec_context = node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]

            attrs_node = {}
            attrs_node['label'] = str(node[PTNNCFGraph.ID_NODE_ATTR]) + ' ' + str(op_exec_context.input_agnostic)

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
                          nx_node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR],
                          nx_node)

    def find_node_in_nx_graph_by_scope(self, scope: 'Scope') -> Optional[dict]:
        """
        Looking for node with scope == scope in networkx graph.
        :param self: graphs to work on
        :param scope: Scope to find in graph
        :return: node from networkx graph for graph (or None if such scope is not found)
        """
        nodes = self._nx_graph.nodes
        for node_key in nodes:
            if nodes[node_key][PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model == scope:
                return nodes[node_key]
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
            if nodes[node_key][PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic == input_agnostic:
                return nodes[node_key]
        return None

    def get_op_nodes_in_scope(self, scope: 'Scope') -> List:
        matching_graph_op_nodes = []
        for _, node in self._nx_graph.nodes.items():
            op_scope = node[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic.scope_in_model
            if op_scope in scope:
                matching_graph_op_nodes.append(node)
        return matching_graph_op_nodes


class NNCFNodeExpression(NodeExpression):
    def __init__(self, node_type: str = None, filter_fn=None):
        super().__init__(node_type, filter_fn, node_type_fn=PTNNCFGraph.node_type_fn)


def pt_get_module_identifier(node: PTNNCFNode):
    return str(node.op_exec_context.scope_in_model)
