# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from typing import Any, Dict, Generator, List, Optional, Tuple

import networkx as nx
import networkx.algorithms.isomorphism as iso
import torch
from torch import Tensor

from nncf import nncf_logger
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta
from nncf.torch.dynamic_graph.trace_tensor import TracedParameter
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor


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
    def __call__(
        self, node_inputs: List[TensorMeta], real_inputs: List[TensorMeta], tm_comparators: List[TensorMetaComparator]
    ) -> bool:
        raise NotImplementedError


class FirstInputsMatcher(InputsMatcher):
    def __call__(
        self, node_inputs: List[TensorMeta], real_inputs: List[TensorMeta], tm_comparators: List[TensorMetaComparator]
    ) -> bool:
        if not node_inputs or not real_inputs:
            return False

        if not node_inputs[0] or not real_inputs[0]:
            return False

        for tm_comparator in tm_comparators:
            if not tm_comparator(node_inputs[0], real_inputs[0]):
                return False
        return True


class DefaultInputsMatcher(InputsMatcher):
    def __call__(
        self,
        saved_inputs: List[TensorMeta],
        actual_inputs: List[TensorMeta],
        tm_comparators: List[TensorMetaComparator],
    ) -> bool:
        if saved_inputs is None and actual_inputs:
            return False

        matched_with_unexpected_tensors = False
        for saved_input, actual_input in zip(saved_inputs, actual_inputs):
            if saved_input is None and actual_input is None:
                continue
            if (saved_input is None) and (actual_input is not None):
                # torch.Tensor.size() seems to return ints when not tracing ONNX
                # and tensors when tracing ONNX. This breaks input-based node matching whenever
                # torch.Tensor.size() return value is passed into a NNCF-traced operation (such as `view`)
                # because at graph building time it expected to see ints as args and now it sees tensors.
                # To mitigate this, will only match inputs against the positions which had tensors during build-time
                # and disregard the rest of the argument positions.
                matched_with_unexpected_tensors = True
                continue
            if (saved_input is not None) and (actual_input is None):
                return False
            for tm_comparator in tm_comparators:
                if not tm_comparator(saved_input, actual_input):
                    return False
        if matched_with_unexpected_tensors:
            nncf_logger.debug(
                f"Had to match a node to an op which has tensors at positions where there were "
                f"no tensors at graph building time:\n"
                f"Node input metas: {saved_inputs}, but op input metas: {actual_inputs}"
            )
        return True


class OperationExecutionContext:
    """
    Information that allows to uniquely identify an operation inside the NNCF graph,
    i.e. determine whether an execution of the operator inside the module has already been
    registered as a node in the graph or not (in the latter case a new node would have to
    be created
    """

    def __init__(
        self,
        operator_name: str,
        scope_in_model: Scope,
        call_order: int,
        tensor_metas: List[TensorMeta],
        tm_comparators: List[TensorMetaComparator] = None,
        input_matcher: InputsMatcher = None,
    ):
        self.op_address = OperationAddress(operator_name, scope_in_model, call_order)
        # This should be a list with a length equal to the number of inputs.
        # "None" values in this list correspond to non-tensor input nodes.
        self.tensor_metas = tensor_metas
        self.tm_comparators = tm_comparators if tm_comparators else [DefaultTensorMetaComparator()]
        self.input_matcher = input_matcher if input_matcher else DefaultInputsMatcher()

    def __eq__(self, other):
        return self.op_address == other.op_address and Counter(self.tensor_metas) == Counter(other.tensor_metas)

    def matches_saved_inputs_from(self, other: "OperationExecutionContext"):
        # WARNING: not commutative
        return self.op_address == other.op_address and self.input_matcher(
            other.tensor_metas, self.tensor_metas, self.tm_comparators
        )

    def __hash__(self):
        return hash((self.operator_name, tuple(self.scope_in_model), self.call_order, tuple(self.tensor_metas)))

    def __str__(self):
        input_info_str = ""
        for meta in self.tensor_metas:
            if meta is None:
                input_info_str += "N;"
            else:
                input_info_str += str(meta) + ";"

        return super().__str__() + "(" + input_info_str + ")"

    @property
    def operator_name(self):
        return self.op_address.operator_name

    @property
    def scope_in_model(self) -> Scope:
        return self.op_address.scope_in_model

    @property
    def call_order(self):
        return self.op_address.call_order


class DynamicGraphNodeParameters:
    def __init__(
        self,
        layer_attributes: BaseLayerAttributes,
        ignored_algorithms: List[str],
        is_called_inside_nncf_module: bool,
        calling_module_id: int,
    ):
        self.layer_attributes = layer_attributes
        self.ignored_algorithms = ignored_algorithms
        self.is_called_inside_nncf_module = is_called_inside_nncf_module
        self.calling_module_id = calling_module_id


class DynamicGraphNode:
    def __init__(
        self,
        node_id: int,
        node_key: str,
        layer_attributes: BaseLayerAttributes,
        op_exec_context: OperationExecutionContext,
        calling_module_id: int,
        ignored_algorithms: List[str],
        is_called_inside_nncf_module: bool,
        is_in_iteration_scope: bool,
    ):
        self.node_id = node_id
        self.node_key = node_key
        self.layer_attributes = layer_attributes
        self.op_exec_context = op_exec_context
        self.calling_module_id = calling_module_id
        self.ignored_algorithms = ignored_algorithms
        self.is_called_inside_nncf_module = is_called_inside_nncf_module
        self.is_in_iteration_scope = is_in_iteration_scope

    @classmethod
    def build_from_nx_node(cls, nx_node: Dict[str, Any]) -> "DynamicGraphNode":
        return cls(
            node_id=nx_node[DynamicGraph.ID_NODE_ATTR],
            node_key=nx_node[DynamicGraph.KEY_NODE_ATTR],
            layer_attributes=nx_node.get(DynamicGraph.LAYER_ATTRIBUTES),
            op_exec_context=nx_node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR],
            ignored_algorithms=nx_node[DynamicGraph.IGNORED_ALGOS_NODE_ATTR],
            is_called_inside_nncf_module=nx_node[DynamicGraph.IS_CALLED_INSIDE_NNCF_MODULE],
            is_in_iteration_scope=nx_node[DynamicGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR],
            calling_module_id=nx_node[DynamicGraph.CALLING_MODULE_ID],
        )

    def __eq__(self, other: "DynamicGraphNode") -> bool:
        return self.__dict__ == other.__dict__

    def __str__(self):
        return self.node_key


class DynamicGraphEdge:
    def __init__(
        self,
        from_node_id: int,
        to_node_id: int,
        activation_shape: List[int],
        input_port_id: int,
        output_port_id: int,
        dtype: Dtype,
        parallel_input_port_ids: List[int],
    ):
        """
        :param from_node_id - A numeric identifier of the starting node of the edge
        :param to_node_id - A numeric identifier of the node to which the edge is pointing
        :param activation_shape - The shape of the tensor associated with this edge
        :param input_port_id - Port of the operation consuming the tensor that this edge
          represents.
        :param output_port_id - Port of the operation that produced the tensor represented by this edge.
        :param dtype - Data type of the tensor represented by this edge.
        :param edge_multiplicity - Multiplicity of the edge. This is a workaround the the DiGraph's
          requirement to have no more than 1 edge between any two nodes, while, for instance, an operation
          of multiplying a tensor with itself, for instance, needs to somehow store in the graph
          the information that the operation actually has two inputs, even though one and the same
          tensor (i.e. edge) is used as both inputs; in this case `edge_multiplicity` must be set to 2.
        """
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.activation_shape = activation_shape
        self.input_port_id = input_port_id
        self.output_port_id = output_port_id
        self.dtype = dtype
        self.parallel_input_port_ids = parallel_input_port_ids

    @classmethod
    def build_between_two_nx_nodes(
        cls, from_nx_node: Dict[str, Any], to_nx_node: Dict[str, Any], nx_edge: Dict[str, Any]
    ) -> "DynamicGraphEdge":
        from_node_id = from_nx_node[DynamicGraph.ID_NODE_ATTR]
        to_node_id = to_nx_node[DynamicGraph.ID_NODE_ATTR]
        return DynamicGraphEdge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            activation_shape=nx_edge[DynamicGraph.ACTIVATION_SHAPE_EDGE_ATTR],
            input_port_id=nx_edge[DynamicGraph.INPUT_PORT_ID_EDGE_ATTR],
            output_port_id=nx_edge[DynamicGraph.OUTPUT_PORT_ID_EDGE_ATTR],
            dtype=nx_edge[DynamicGraph.ACTIVATION_DTYPE_EDGE_ATTR],
            parallel_input_port_ids=nx_edge[DynamicGraph.PARALLEL_INPUT_PORT_IDS],
        )


class DefaultScopeNodeMatcher:
    def __init__(self, node_id_to_key_dict, nx_graph):
        self._node_id_to_key_dict = node_id_to_key_dict
        self._nx_graph = nx_graph
        self._inputless_nodes: Dict[str, DynamicGraphNode] = {}

    def get_node_by_id(self, node_id):
        return self._nx_graph.nodes[self._node_id_to_key_dict[node_id]]

    def _find_nodes_with_matching_context_among_inputless(
        self, op_exec_context: OperationExecutionContext
    ) -> Dict[str, DynamicGraphNode]:
        node_candidates = {}
        for nx_node_key, node in self._inputless_nodes.items():
            if op_exec_context.matches_saved_inputs_from(node.op_exec_context):
                node_candidates[nx_node_key] = node
        return node_candidates

    def _find_nodes_with_matching_context_and_inputs(
        self, op_exec_context: OperationExecutionContext
    ) -> Dict[str, DynamicGraphNode]:
        nx_node_candidates = {}
        for info in op_exec_context.tensor_metas:
            if info is None or info.creator_id is None:
                continue
            creator_id = info.creator_id
            for successor_node_key in self._nx_graph.successors(self._node_id_to_key_dict[creator_id]):
                successor_node = self._nx_graph.nodes[successor_node_key]
                if op_exec_context.matches_saved_inputs_from(successor_node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR]):
                    nx_node_candidates[successor_node_key] = successor_node

        node_candidates: Dict[str, DynamicGraphNode] = {}
        for nx_node_key, nx_node_dict in nx_node_candidates.items():
            node_candidates[nx_node_key] = DynamicGraphNode.build_from_nx_node(nx_node_dict)

        return node_candidates

    def add_node(
        self,
        op_exec_context: OperationExecutionContext,
        inputs,
        node_parameters: DynamicGraphNodeParameters,
        is_in_iteration_scope: bool = False,
    ) -> DynamicGraphNode:
        node_id = len(self._node_id_to_key_dict)

        name_parts = (str(op_exec_context.scope_in_model), op_exec_context.operator_name)
        node_key = "{idx} {uri}".format(uri="/".join(name_parts), idx=node_id)

        nncf_logger.debug(f"New node added to NNCF graph: {node_key}")

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            DynamicGraph.ID_NODE_ATTR: node_id,
            DynamicGraph.KEY_NODE_ATTR: node_key,
            DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR: op_exec_context,
            DynamicGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR: is_in_iteration_scope,
            DynamicGraph.CALLING_MODULE_ID: node_parameters.calling_module_id,
        }
        if node_parameters.layer_attributes is not None:
            attrs[DynamicGraph.LAYER_ATTRIBUTES] = node_parameters.layer_attributes

        if node_parameters.ignored_algorithms is not None:
            attrs[DynamicGraph.IGNORED_ALGOS_NODE_ATTR] = node_parameters.ignored_algorithms
        else:
            attrs[DynamicGraph.IGNORED_ALGOS_NODE_ATTR] = []
        attrs[DynamicGraph.IS_CALLED_INSIDE_NNCF_MODULE] = node_parameters.is_called_inside_nncf_module

        self._nx_graph.add_node(node_key, **attrs)

        has_traced_inputs = False
        for i, info in enumerate(op_exec_context.tensor_metas):
            input_port_id = i
            if info is None or info.creator_id is None:
                continue

            has_traced_inputs = True
            parent = self._node_id_to_key_dict[info.creator_id]
            if self._nx_graph.get_edge_data(parent, node_key) is not None:
                self._nx_graph.edges[parent, node_key][DynamicGraph.PARALLEL_INPUT_PORT_IDS] += [input_port_id]
                continue
            self._nx_graph.add_edge(parent, node_key)
            self._nx_graph.edges[parent, node_key][DynamicGraph.ACTIVATION_SHAPE_EDGE_ATTR] = info.shape
            self._nx_graph.edges[parent, node_key][DynamicGraph.INPUT_PORT_ID_EDGE_ATTR] = input_port_id
            self._nx_graph.edges[parent, node_key][DynamicGraph.OUTPUT_PORT_ID_EDGE_ATTR] = info.index
            self._nx_graph.edges[parent, node_key][DynamicGraph.ACTIVATION_DTYPE_EDGE_ATTR] = info.dtype
            self._nx_graph.edges[parent, node_key][DynamicGraph.PARALLEL_INPUT_PORT_IDS] = []

        nx_node_dict = self._nx_graph.nodes[node_key]
        node = DynamicGraphNode.build_from_nx_node(nx_node_dict)

        if not has_traced_inputs:
            self._inputless_nodes[node_key] = node

        return node

    def find_node(
        self, op_address: OperationAddress, tensor_metas: List[TensorMeta], tm_comparators: List[TensorMetaComparator]
    ) -> DynamicGraphNode:
        op_exec_context = OperationExecutionContext(
            op_address.operator_name,
            op_address.scope_in_model,
            op_address.call_order,
            tensor_metas,
            tm_comparators=tm_comparators,
        )
        node_candidates = self._find_nodes_with_matching_context_and_inputs(op_exec_context)
        if not node_candidates:
            node_candidates = self._find_nodes_with_matching_context_among_inputless(op_exec_context)

        node_candidates = list(node_candidates.values())
        result = None
        if len(node_candidates) == 1:
            result = node_candidates[0]
        if len(node_candidates) > 1:
            nncf_logger.debug(f"More than one node was matched against context {op_exec_context}")
            result = node_candidates[0]

        return result


class IterationScopeNodeMatcher(DefaultScopeNodeMatcher):
    def __init__(self, node_id_to_key_dict, nx_graph):
        super().__init__(node_id_to_key_dict, nx_graph)
        self._first_iteration_nodes: {str: {str: DynamicGraphNode}} = {}

    def save_first_iteration_node(self, inputs: OperatorInput, node: DynamicGraphNode):
        """
        It finds and saves "starting" points of iteration for further matching with them on next iteration,
        instead of adding new nodes for each iteration. "Starting" points of iteration are nodes
            * that have at least one input node, which is outside of iteration scope
            * or whose all inputs are not tensors with TracedTensorMixin tracing capabilities.
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
                traced_parameter_inputs = []
                non_tensor_inputs = []
                for i in inputs:
                    input_obj = i.getter()
                    if isinstance(input_obj, Tensor):
                        if isinstance(input_obj, TracedTensor):
                            traced_tensor_inputs.append(input_obj)
                        elif isinstance(input_obj, TracedParameter):
                            traced_parameter_inputs.append(input_obj)
                        else:
                            untraced_tensor_inputs.append(input_obj)
                    else:
                        non_tensor_inputs.append(input_obj)

                for i in traced_parameter_inputs:
                    if i.tensor_meta is not None:
                        traced_tensor_inputs.append(i)

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
                    nncf_logger.debug(f"Found first iteration node: {name} in scope: {iter_scope}")

    def add_node(
        self,
        op_exec_context: OperationExecutionContext,
        inputs,
        node_parameters: DynamicGraphNodeParameters,
        is_in_iteration_scope: bool = True,
    ) -> DynamicGraphNode:
        node = super().add_node(op_exec_context, inputs, node_parameters, is_in_iteration_scope=True)
        self.save_first_iteration_node(inputs, node)
        return node

    def find_node(
        self, op_address: OperationAddress, tensor_metas: List[TensorMeta], tm_comparators: List[TensorMetaComparator]
    ) -> Optional[DynamicGraphNode]:
        iter_scopes = op_address.scope_in_model.get_iteration_scopes()
        # compare meta information about first input nodes during the matching. During the iteration some nodes may
        # change number of inputs, e.g. on concat of hidden outputs
        input_matcher = FirstInputsMatcher()
        op_exec_context = OperationExecutionContext(
            op_address.operator_name,
            op_address.scope_in_model,
            op_address.call_order,
            tensor_metas,
            input_matcher=input_matcher,
            tm_comparators=tm_comparators,
        )
        node_candidates = self._find_nodes_with_matching_context_and_inputs(op_exec_context)
        if not node_candidates:
            op_exec_context = OperationExecutionContext(
                op_address.operator_name,
                op_address.scope_in_model,
                op_address.call_order,
                tensor_metas,
                tm_comparators=tm_comparators,
            )
            node_candidates = self._find_nodes_with_matching_context_among_inputless(op_exec_context)
            if not node_candidates and iter_scopes:
                # ignore information about node creator and index of input
                comparators = tm_comparators + [ShapeOnlyTensorMetaComparator()]
                op_exec_context = OperationExecutionContext(
                    op_address.operator_name,
                    op_address.scope_in_model,
                    op_address.call_order,
                    tensor_metas,
                    tm_comparators=comparators,
                )
                # match with starting points of iteration
                iter_nodes = self._match_first_iteration_nodes(op_exec_context, iter_scopes)
                for node_key, node in iter_nodes.items():
                    node_candidates[node_key] = node

        node_candidates = list(node_candidates.values())
        result = None
        if len(node_candidates) == 1:
            result = node_candidates[0]
        if len(node_candidates) > 1:
            nncf_logger.debug(f"More than one node was matched against context {op_exec_context}")
            result = node_candidates[0]

        return result

    def _match_first_iteration_nodes(self, op_exec_context: OperationExecutionContext, iter_scopes):
        node_candidates = {}
        for iter_scope in iter_scopes:
            if iter_scope in self._first_iteration_nodes:
                for name, node in self._first_iteration_nodes[iter_scope].items():
                    if op_exec_context.matches_saved_inputs_from(node.op_exec_context):
                        node_candidates[name] = node
                        break
                if node_candidates:
                    break
        return node_candidates

    def get_first_iteration_modules(self) -> Dict:
        return self._first_iteration_nodes


class NodeManager:
    def __init__(self, node_id_to_key_dict, nx_graph):
        self.base_matcher = DefaultScopeNodeMatcher(node_id_to_key_dict, nx_graph)
        self.iteration_matcher = IterationScopeNodeMatcher(node_id_to_key_dict, nx_graph)

    # TODO: optimize by matching exact module type
    @staticmethod
    def _within_iteration(scope: Scope):
        from nncf.torch.layers import ITERATION_MODULES

        for scope_element in scope.scope_elements:
            if scope_element.calling_module_class_name in ITERATION_MODULES.registry_dict:
                return True
        return False

    def choose_matcher(self, op_address: OperationAddress) -> DefaultScopeNodeMatcher:
        if self._within_iteration(op_address.scope_in_model):
            return self.iteration_matcher
        return self.base_matcher

    @staticmethod
    def choose_tm_comparators(
        op_address: OperationAddress, input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]]
    ) -> List[TensorMetaComparator]:
        result = []
        for pairs in input_comparators_per_scope:
            comparator, scopes = pairs
            for scope in scopes:
                if scope in str(op_address):
                    result.append(comparator)
        return result

    def find_node(
        self,
        op_address: OperationAddress,
        tensor_metas: List[TensorMeta],
        input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
    ) -> DynamicGraphNode:
        matcher = self.choose_matcher(op_address)
        comparators = self.choose_tm_comparators(op_address, input_comparators_per_scope)
        return matcher.find_node(op_address, tensor_metas, comparators)

    def add_node(
        self,
        op_address: OperationAddress,
        tensor_metas: List[TensorMeta],
        tm_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
        inputs,
        node_parameters: DynamicGraphNodeParameters,
    ) -> DynamicGraphNode:
        matcher = self.choose_matcher(op_address)
        tm_comparators = self.choose_tm_comparators(op_address, tm_comparators_per_scope)
        op_exec_context = OperationExecutionContext(
            op_address.operator_name,
            op_address.scope_in_model,
            op_address.call_order,
            tensor_metas,
            tm_comparators=tm_comparators,
        )

        return matcher.add_node(op_exec_context, inputs, node_parameters)


class DynamicGraph:
    """
    The class for representing a graph dynamically built during a PyTorch model's `forward` method execution
    within a nncf.torch.dynamic_graph.context.TracingContext. This graph may change from a forward call to a
    forward call if the execution paths of the model change between the calls - this sets DynamicGraph apart from
    NNCFGraph which is a static representation of the model's structure. The DynamicGraph has limited support for
    RNN tracing and is rather suited to regular DNN tracing.
    """

    ID_NODE_ATTR = "id"
    KEY_NODE_ATTR = "key"
    LAYER_ATTRIBUTES = "layer_attributes"
    OP_EXEC_CONTEXT_NODE_ATTR = "op_exec_context"
    ACTIVATION_SHAPE_EDGE_ATTR = "activation_shape"
    ACTIVATION_DTYPE_EDGE_ATTR = "activation_dtype"
    INPUT_PORT_ID_EDGE_ATTR = "input_port_id"
    OUTPUT_PORT_ID_EDGE_ATTR = "output_port_id"
    IGNORED_ALGOS_NODE_ATTR = "ignored_algos"
    IS_CALLED_INSIDE_NNCF_MODULE = "is_called_inside_nncf_module"
    IS_IN_ITERATION_SCOPE_NODE_ATTR = "is_in_iteration_scope"
    CALLING_MODULE_ID = "calling_module_id"
    PARALLEL_INPUT_PORT_IDS = "parallel_input_port_ids"

    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict = {}
        self.match_manager = NodeManager(self._node_id_to_key_dict, self._nx_graph)
        self._input_nncf_nodes = []
        self._output_nncf_nodes = []
        self._integer_input_nodes = []

    def __eq__(self, other: "DynamicGraph"):
        nm = iso.categorical_node_match(
            [
                DynamicGraph.ID_NODE_ATTR,
                DynamicGraph.KEY_NODE_ATTR,
                DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR,
                DynamicGraph.LAYER_ATTRIBUTES,
                DynamicGraph.CALLING_MODULE_ID,
            ],
            [None, None, None, None, None],
        )
        em = iso.categorical_edge_match(
            [DynamicGraph.ACTIVATION_SHAPE_EDGE_ATTR, DynamicGraph.INPUT_PORT_ID_EDGE_ATTR], [None, None]
        )
        return nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)

    def find_node(
        self,
        op_address: OperationAddress,
        tensor_metas: List[TensorMeta],
        input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
    ) -> DynamicGraphNode:
        return self.match_manager.find_node(op_address, tensor_metas, input_comparators_per_scope)

    def add_node(
        self,
        op_address: OperationAddress,
        tensor_metas: List[TensorMeta],
        input_comparators_per_scope: List[Tuple[TensorMetaComparator, List[str]]],
        inputs: OperatorInput,
        node_parameters: DynamicGraphNodeParameters,
    ) -> DynamicGraphNode:
        node = self.match_manager.add_node(
            op_address, tensor_metas, input_comparators_per_scope, inputs, node_parameters
        )

        from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
        from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME

        if node.op_exec_context.operator_name == MODEL_INPUT_OP_NAME:
            self._input_nncf_nodes.append(node)
            # Currently the MODEL_INPUT_OP_NAME node is added when an input is wrapped as
            # _ = nncf_model_input(input_tensor)
            # so it is expected that there 0-th positional arg will be the torch.Tensor we need to inspect
            tensor_input = inputs.op_args[0]
            assert isinstance(tensor_input, torch.Tensor)
            if tensor_input.dtype in (torch.int32, torch.int64, torch.long):
                self._integer_input_nodes.append(node)

        if node.op_exec_context.operator_name == MODEL_OUTPUT_OP_NAME:
            self._output_nncf_nodes.append(node)
        return node

    def get_input_nodes(self) -> List[DynamicGraphNode]:
        return self._input_nncf_nodes

    def is_integer_input_node(self, node: DynamicGraphNode) -> bool:
        return node in self._integer_input_nodes

    def get_output_nodes(self) -> List[DynamicGraphNode]:
        return self._output_nncf_nodes

    def get_nodes_count(self) -> int:
        return self._nx_graph.number_of_nodes()

    def get_all_nodes(self) -> List[DynamicGraphNode]:
        all_nodes = []
        for node_key in self._node_id_to_key_dict.values():
            dynamic_graph_node = DynamicGraphNode.build_from_nx_node(self._nx_graph.nodes[node_key])
            all_nodes.append(dynamic_graph_node)
        return all_nodes

    def get_all_edges(self) -> Generator[DynamicGraphEdge, None, None]:
        """
        Generates all edges in the graph
        """
        for from_nx_node_key, to_nx_node_key in self._nx_graph.in_edges:
            yield self._get_edge(self._get_node_by_key(from_nx_node_key), self._get_node_by_key(to_nx_node_key))

    def get_input_edges(self, node: DynamicGraphNode) -> List[DynamicGraphEdge]:
        """
        Returns edges of input tensors with description sorted by input port ID.

        :param node: Consumer node.
        :return: List of input edges for the node sorted by input port ID.
        """
        input_nodes = self._get_previous_nodes(node)
        edges = [self._get_edge(from_node, node) for from_node in input_nodes]
        return sorted(edges, key=lambda x: x.input_port_id)

    def set_layer_attributes_to_node(self, node: DynamicGraphNode, layer_attributes: BaseLayerAttributes):
        self._nx_graph.nodes[node.node_key][DynamicGraph.LAYER_ATTRIBUTES] = layer_attributes

    def is_graph_with_iteration_modules(self) -> bool:
        return len(self.match_manager.iteration_matcher.get_first_iteration_modules()) > 0

    def _get_node_by_key(self, key: str) -> DynamicGraphNode:
        """
        :param key: key (node_name) of the node.
        :return: DynamicGraphNode in a graph with such key.
        """
        return DynamicGraphNode.build_from_nx_node(self._nx_graph.nodes[key])

    def _get_previous_nodes(self, node: DynamicGraphNode) -> List[DynamicGraphNode]:
        nx_node_keys = self._nx_graph.pred[self._node_id_to_key_dict[node.node_id]]
        return [DynamicGraphNode.build_from_nx_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def _get_edge(self, from_node: DynamicGraphNode, to_node: DynamicGraphNode) -> DynamicGraphEdge:
        nx_edge = self._get_nx_edge(from_node, to_node)
        from_nx_node = self._nx_graph.nodes[from_node.node_key]
        to_nx_node = self._nx_graph.nodes[to_node.node_key]
        return DynamicGraphEdge.build_between_two_nx_nodes(from_nx_node, to_nx_node, nx_edge)

    def _get_nx_edge(self, node_u: DynamicGraphNode, node_v: DynamicGraphNode):
        nx_node_u = self._nx_graph.nodes[self._node_id_to_key_dict[node_u.node_id]]
        nx_node_v = self._nx_graph.nodes[self._node_id_to_key_dict[node_v.node_id]]
        return self._nx_graph.edges[nx_node_u["key"], nx_node_v["key"]]
