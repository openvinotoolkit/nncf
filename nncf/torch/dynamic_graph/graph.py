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

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import networkx as nx
import networkx.algorithms.isomorphism as iso
from torch import Tensor

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.operation_address import OperationAddress


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


class OperationExecutionContext:
    """
    Information that allows to uniquely identify an operation inside the NNCF graph,
    i.e. determine whether an execution of the operator inside the module has already been
    registered as a node in the graph or not (in the latter case a new node would have to
    be created
    """

    def __init__(self,
                 operator_name: str,
                 scope_in_model: Scope,
                 call_order: int,
                 tensor_metas: List[TensorMeta],
                 tm_comparators: List[TensorMetaComparator] = None,
                 input_matcher: InputsMatcher = None):
        self.op_address = OperationAddress(operator_name, scope_in_model, call_order)
        # This should be a list with a length equal to the number of inputs.
        # "None" values in this list correspond to non-tensor input nodes.
        self.tensor_metas = tensor_metas
        self.tm_comparators = tm_comparators if tm_comparators else [
            DefaultTensorMetaComparator()]
        self.input_matcher = input_matcher if input_matcher else DefaultInputsMatcher()

    def __eq__(self, other: 'OperationExecutionContext'):
        return (self.op_address == other.op_address) and \
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
        return self.op_address.operator_name

    @property
    def scope_in_model(self) -> Scope:
        return self.op_address.scope_in_model

    @property
    def call_order(self):
        return self.op_address.call_order


class DynamicGraphNode:
    def __init__(self, node_id: int, node_key: str, layer_attributes: BaseLayerAttributes,
                 op_exec_context: OperationExecutionContext, ignored_algorithms: List[str],
                 is_in_iteration_scope: bool):
        self.node_id = node_id
        self.node_key = node_key
        self.layer_attributes = layer_attributes
        self.op_exec_context = op_exec_context
        self.ignored_algorithms = ignored_algorithms
        self.is_in_iteration_scope = is_in_iteration_scope

    def __eq__(self, other: 'DynamicGraphNode') -> bool:
        return self.__dict__ == other.__dict__

    def __str__(self):
        return self.node_key


class DynamicGraphEdge:
    def __init__(self, from_node_id: int, to_node_id: int,
                 activation_shape: List[int], input_port_id: int):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.activation_shape = activation_shape
        self.input_port_id = input_port_id


class DefaultScopeNodeMatcher:
    def __init__(self, node_id_to_key_dict, nx_graph):
        self._node_id_to_key_dict = node_id_to_key_dict
        self._nx_graph = nx_graph
        self._inputless_nodes = dict()  # type: Dict[str, DynamicGraphNode]

    def get_node_by_id(self, node_id):
        return self._nx_graph.nodes[self._node_id_to_key_dict[node_id]]

    def _find_nodes_with_matching_context_among_inputless(self, op_exec_context: OperationExecutionContext) \
            -> Dict[str, DynamicGraphNode]:
        node_candidates = {}
        for nx_node_key, node in self._inputless_nodes.items():
            if node.op_exec_context == op_exec_context:
                node_candidates[nx_node_key] = node
        return node_candidates

    def _find_nodes_with_matching_context_and_inputs(self, op_exec_context: OperationExecutionContext) \
            -> Dict[str, DynamicGraphNode]:
        nx_node_candidates = {}
        for info in op_exec_context.tensor_metas:
            if info is None or info.creator_id is None:
                continue
            creator_id = info.creator_id
            for successor_node_key in self._nx_graph.successors(self._node_id_to_key_dict[creator_id]):
                successor_node = self._nx_graph.nodes[successor_node_key]
                if op_exec_context == successor_node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                    nx_node_candidates[successor_node_key] = successor_node

        node_candidates = {}  # type: Dict[str, DynamicGraphNode]
        for nx_node_key, nx_node in nx_node_candidates.items():
            node_candidates[nx_node_key] = DynamicGraphNode(
                node_id=nx_node[DynamicGraph.ID_NODE_ATTR],
                node_key=nx_node[DynamicGraph.KEY_NODE_ATTR],
                layer_attributes=nx_node.get(DynamicGraph.LAYER_ATTRIBUTES),
                op_exec_context=nx_node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR],
                ignored_algorithms=nx_node[DynamicGraph.IGNORED_ALGOS_NODE_ATTR],
                is_in_iteration_scope=nx_node[DynamicGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR])

        return node_candidates

    def add_node(self, op_exec_context: OperationExecutionContext, inputs,
                 layer_attrs: BaseLayerAttributes = None,
                 ignored_algorithms: List[str] = None,
                 is_in_iteration_scope: bool = False) -> DynamicGraphNode:
        node_id = len(self._node_id_to_key_dict)

        name_parts = (str(op_exec_context.scope_in_model), op_exec_context.operator_name)
        node_key = '{idx} {uri}'.format(uri='/'.join(name_parts), idx=node_id)

        nncf_logger.debug("New node added to NNCF graph: {}".format(node_key))

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            DynamicGraph.ID_NODE_ATTR: node_id,
            DynamicGraph.KEY_NODE_ATTR: node_key,
            DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR: op_exec_context,
            DynamicGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR: is_in_iteration_scope
        }
        if layer_attrs is not None:
            attrs[DynamicGraph.LAYER_ATTRIBUTES] = layer_attrs

        if ignored_algorithms is not None:
            attrs[DynamicGraph.IGNORED_ALGOS_NODE_ATTR] = ignored_algorithms
        else:
            attrs[DynamicGraph.IGNORED_ALGOS_NODE_ATTR] = []

        self._nx_graph.add_node(node_key, **attrs)

        has_traced_inputs = False
        for i, info in enumerate(op_exec_context.tensor_metas):
            if info is None or info.creator_id is None:
                continue
            parent = self._node_id_to_key_dict[info.creator_id]
            self._nx_graph.add_edge(parent, node_key)
            has_traced_inputs = True
            self._nx_graph.edges[parent, node_key][DynamicGraph.ACTIVATION_SHAPE_EDGE_ATTR] = info.shape
            self._nx_graph.edges[parent, node_key][DynamicGraph.IN_PORT_NAME_EDGE_ATTR] = i

        nx_node_dict = self._nx_graph.nodes[node_key]
        node = DynamicGraphNode(node_id=nx_node_dict[DynamicGraph.ID_NODE_ATTR],
                                node_key=nx_node_dict[DynamicGraph.KEY_NODE_ATTR],
                                layer_attributes=nx_node_dict.get(DynamicGraph.LAYER_ATTRIBUTES),
                                op_exec_context=nx_node_dict[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR],
                                ignored_algorithms=nx_node_dict[DynamicGraph.IGNORED_ALGOS_NODE_ATTR],
                                is_in_iteration_scope=nx_node_dict[DynamicGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR])

        if not has_traced_inputs:
            self._inputless_nodes[node_key] = node

        return node

    def find_node(self, op_address: OperationAddress,
                  tensor_metas: List[TensorMeta],
                  tm_comparators: List[TensorMetaComparator]) -> DynamicGraphNode:
        op_exec_context = OperationExecutionContext(op_address.operator_name,
                                                    op_address.scope_in_model,
                                                    op_address.call_order,
                                                    tensor_metas,
                                                    tm_comparators=tm_comparators)
        node_candidates = self._find_nodes_with_matching_context_and_inputs(op_exec_context)
        if not node_candidates:
            node_candidates = self._find_nodes_with_matching_context_among_inputless(op_exec_context)

        node_candidates = list(node_candidates.values())
        result = None
        if len(node_candidates) == 1:
            result = node_candidates[0]
        if len(node_candidates) > 1:
            nncf_logger.warning("More than one node matches input")
            result = node_candidates[0]

        return result


class IterationScopeNodeMatcher(DefaultScopeNodeMatcher):
    def __init__(self, node_id_to_key_dict, nx_graph):
        super().__init__(node_id_to_key_dict, nx_graph)
        self._first_iteration_nodes = {}  # type: {str: {str: DynamicGraphNode}}

    def save_first_iteration_node(self, inputs: 'OperatorInput', node: DynamicGraphNode):
        """
        It finds and saves "starting" points of iteration for further matching with them on next iteration,
        instead of adding new nodes for each iteration. "Starting" points of iteration are nodes
            * that have at least one input node, which is outside of iteration scope
            * or whose all inputs are not TracedTensor
        """
        op_exec_context = node.op_exec_context
        name = str(node)
        iter_scopes = op_exec_context.scope_in_model.get_iteration_scopes()
        if iter_scopes:
            for iter_scope in iter_scopes:
                if iter_scope not in self._first_iteration_nodes:
                    self._first_iteration_nodes[iter_scope] = {}
                first_nodes = self._first_iteration_nodes[iter_scope]
                has_input_outside_iteration = False
                untraced_tensor_inputs = []
                traced_tensor_inputs = []
                non_tensor_inputs = []
                for i in inputs:
                    input_obj = i.getter()
                    if isinstance(input_obj, Tensor):
                        if not isinstance(input_obj, TracedTensor):
                            untraced_tensor_inputs.append(input_obj)
                        else:
                            traced_tensor_inputs.append(input_obj)
                    else:
                        non_tensor_inputs.append(input_obj)

                for i in traced_tensor_inputs:
                    creator_id = i.tensor_meta.creator_id
                    creator_node = self.get_node_by_id(creator_id)
                    creator_node_op_exec_ctx = creator_node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                    within_scopes = creator_node_op_exec_ctx.scope_in_model.get_iteration_scopes()
                    if iter_scope not in within_scopes:
                        has_input_outside_iteration = True

                if len(untraced_tensor_inputs) == (len(inputs) - len(non_tensor_inputs)):
                    has_input_outside_iteration = True
                if has_input_outside_iteration:
                    node_name = str(op_exec_context.op_address)
                    first_nodes[node_name] = node
                    nncf_logger.debug('Found first iteration node: {} in scope: {}'.format(name, iter_scope))

    def add_node(self, op_exec_context: OperationExecutionContext, inputs,
                 layer_attrs: BaseLayerAttributes = None,
                 ignored_algorithms: List[str] = None,
                 is_in_iteration_scope: bool = True) -> DynamicGraphNode:
        node = super().add_node(op_exec_context, inputs, layer_attrs, ignored_algorithms,
                                is_in_iteration_scope=True)
        self.save_first_iteration_node(inputs, node)
        return node

    def find_node(self,
                  op_address: OperationAddress,
                  tensor_metas: List[TensorMeta],
                  tm_comparators: List[TensorMetaComparator]) -> Optional[DynamicGraphNode]:
        iter_scopes = op_address.scope_in_model.get_iteration_scopes()
        # compare meta information about first input nodes during the matching. During the iteration some nodes may
        # change number of inputs, e.g. on concat of hidden outputs
        input_matcher = FirstInputsMatcher()
        op_exec_context = OperationExecutionContext(op_address.operator_name,
                                                    op_address.scope_in_model,
                                                    op_address.call_order,
                                                    tensor_metas,
                                                    input_matcher=input_matcher,
                                                    tm_comparators=tm_comparators)
        node_candidates = self._find_nodes_with_matching_context_and_inputs(op_exec_context)
        if not node_candidates:
            op_exec_context = OperationExecutionContext(op_address.operator_name,
                                                        op_address.scope_in_model,
                                                        op_address.call_order,
                                                        tensor_metas,
                                                        tm_comparators=tm_comparators)
            node_candidates = self._find_nodes_with_matching_context_among_inputless(op_exec_context)
            if not node_candidates and iter_scopes:
                # ignore information about node creator and index of input
                comparators = tm_comparators + [ShapeOnlyTensorMetaComparator()]
                op_exec_context = OperationExecutionContext(op_address.operator_name,
                                                            op_address.scope_in_model,
                                                            op_address.call_order,
                                                            tensor_metas,
                                                            tm_comparators=comparators)
                # match with starting points of iteration
                iter_nodes = self._match_first_iteration_nodes(op_exec_context, iter_scopes)
                for node_key, node in iter_nodes.items():
                    node_candidates[node_key] = node

        node_candidates = list(node_candidates.values())
        result = None
        if len(node_candidates) == 1:
            result = node_candidates[0]
        if len(node_candidates) > 1:
            nncf_logger.warning("More than one node matches input")
            result = node_candidates[0]

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
    def __init__(self, node_id_to_key_dict, nx_graph):
        self.base_matcher = DefaultScopeNodeMatcher(node_id_to_key_dict, nx_graph)
        self.iteration_matcher = IterationScopeNodeMatcher(node_id_to_key_dict, nx_graph)

    # TODO: optimize by matching exact module type
    @staticmethod
    def _within_iteration(scope: Scope):
        scope_name = str(scope)
        from nncf.torch.layers import ITERATION_MODULES
        for iter_scope in ITERATION_MODULES.registry_dict:
            if iter_scope in scope_name:
                return True
        return False

    def choose_matcher(self, op_address: OperationAddress) -> DefaultScopeNodeMatcher:
        if self._within_iteration(op_address.scope_in_model):
            return self.iteration_matcher
        return self.base_matcher

    @staticmethod
    def choose_tm_comparators(op_address: OperationAddress,
                              input_comparators_per_scope:
                              List[Tuple[TensorMetaComparator, List[str]]]) -> List[TensorMetaComparator]:
        result = []
        for pairs in input_comparators_per_scope:
            comparator, scopes = pairs
            for scope in scopes:
                if scope in str(op_address):
                    result.append(comparator)
        return result

    def find_node(self, op_address: OperationAddress,
                  tensor_metas: List[TensorMeta],
                  input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]]) -> DynamicGraphNode:
        matcher = self.choose_matcher(op_address)
        comparators = self.choose_tm_comparators(op_address, input_comparators_per_scope)
        return matcher.find_node(op_address, tensor_metas, comparators)

    def add_node(self, op_address: OperationAddress,
                 tensor_metas: List[TensorMeta],
                 tm_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
                 inputs,
                 layer_attrs: BaseLayerAttributes = None,
                 ignored_algorithms: List[str] = None) -> DynamicGraphNode:
        matcher = self.choose_matcher(op_address)
        tm_comparators = self.choose_tm_comparators(op_address, tm_comparators_per_scope)
        op_exec_context = OperationExecutionContext(op_address.operator_name,
                                                    op_address.scope_in_model,
                                                    op_address.call_order,
                                                    tensor_metas,
                                                    tm_comparators=tm_comparators)

        return matcher.add_node(op_exec_context, inputs, layer_attrs, ignored_algorithms)


class DynamicGraph:
    """
    The class for representing a graph dynamically built during a PyTorch model's `forward` method execution
    within a nncf.torch.dynamic_graph.context.TracingContext. This graph may change from a forward call to a
    forward call if the execution paths of the model change between the calls - this sets DynamicGraph apart from
    NNCFGraph which is a static representation of the model's structure. The DynamicGraph has limited support for
    RNN tracing and is rather suited to regular DNN tracing.
    """
    ID_NODE_ATTR = 'id'
    KEY_NODE_ATTR = 'key'
    LAYER_ATTRIBUTES = 'layer_attributes'
    OP_EXEC_CONTEXT_NODE_ATTR = 'op_exec_context'
    ACTIVATION_SHAPE_EDGE_ATTR = 'activation_shape'
    IN_PORT_NAME_EDGE_ATTR = 'in_port'
    IGNORED_ALGOS_NODE_ATTR = 'ignored_algos'
    IS_IN_ITERATION_SCOPE_NODE_ATTR = 'is_in_iteration_scope'

    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict = dict()
        self.match_manager = NodeManager(self._node_id_to_key_dict, self._nx_graph)
        self._input_nncf_nodes = []
        self._output_nncf_nodes = []

    def __eq__(self, other: 'DynamicGraph'):
        nm = iso.categorical_node_match([DynamicGraph.ID_NODE_ATTR,
                                         DynamicGraph.KEY_NODE_ATTR,
                                         DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR,
                                         DynamicGraph.LAYER_ATTRIBUTES], [None, None, None])
        em = iso.categorical_edge_match([DynamicGraph.ACTIVATION_SHAPE_EDGE_ATTR,
                                         DynamicGraph.IN_PORT_NAME_EDGE_ATTR], [None, None])
        return nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)

    def find_node(self,
                  op_address: OperationAddress,
                  tensor_metas: List[TensorMeta],
                  input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]]) -> DynamicGraphNode:
        return self.match_manager.find_node(op_address, tensor_metas, input_comparators_per_scope)

    def add_node(self, op_address: OperationAddress,
                 tensor_metas: List[TensorMeta],
                 input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
                 inputs,
                 layer_attrs: BaseLayerAttributes = None,
                 ignored_algorithms: List[str] = None) -> DynamicGraphNode:
        node = self.match_manager.add_node(op_address, tensor_metas, input_comparators_per_scope, inputs,
                                           layer_attrs, ignored_algorithms)

        from nncf.common.graph import MODEL_OUTPUT_OP_NAME
        from nncf.common.graph import MODEL_INPUT_OP_NAME
        if node.op_exec_context.operator_name == MODEL_INPUT_OP_NAME:
            self._input_nncf_nodes.append(node)

        if node.op_exec_context.operator_name == MODEL_OUTPUT_OP_NAME:
            self._output_nncf_nodes.append(node)
        return node

    def get_input_nodes(self) -> List[DynamicGraphNode]:
        return self._input_nncf_nodes

    def get_output_nodes(self) -> List[DynamicGraphNode]:
        return self._output_nncf_nodes

    def get_nodes_count(self) -> int:
        return self._nx_graph.number_of_nodes()

    def get_all_nodes(self) -> List[DynamicGraphNode]:
        all_nodes = []
        for node_key in self._node_id_to_key_dict.values():
            nx_node = self._nx_graph.nodes[node_key]
            dynamic_graph_node = DynamicGraphNode(
                node_id=nx_node[DynamicGraph.ID_NODE_ATTR],
                node_key=nx_node[DynamicGraph.KEY_NODE_ATTR],
                layer_attributes=nx_node.get(DynamicGraph.LAYER_ATTRIBUTES),
                op_exec_context=nx_node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR],
                ignored_algorithms=nx_node[DynamicGraph.IGNORED_ALGOS_NODE_ATTR],
                is_in_iteration_scope=nx_node[DynamicGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR]
            )
            all_nodes.append(dynamic_graph_node)
        return all_nodes

    def get_all_edges(self) -> List[DynamicGraphEdge]:
        all_edges = []
        for from_node_key, to_node_key in self._nx_graph.edges:
            nx_edge_attrs = self._nx_graph.edges[from_node_key, to_node_key]
            from_node = self._nx_graph.nodes[from_node_key]
            to_node = self._nx_graph.nodes[to_node_key]
            from_node_id = from_node[DynamicGraph.ID_NODE_ATTR]
            to_node_id = to_node[DynamicGraph.ID_NODE_ATTR]
            dynamic_graph_edge = DynamicGraphEdge(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                activation_shape=nx_edge_attrs[DynamicGraph.ACTIVATION_SHAPE_EDGE_ATTR],
                input_port_id=nx_edge_attrs[DynamicGraph.IN_PORT_NAME_EDGE_ATTR])

            all_edges.append(dynamic_graph_edge)
        return all_edges
