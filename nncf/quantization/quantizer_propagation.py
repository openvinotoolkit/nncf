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
# pylint:disable=too-many-lines
from collections import deque
from enum import Enum
from typing import Dict, Tuple, Set, Any, Callable

import networkx as nx
import warnings
from copy import deepcopy

from nncf.dynamic_graph.graph import OperationExecutionContext, NNCFGraph, InputAgnosticOperationExecutionContext
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.dynamic_graph.operator_metatypes import *
from nncf.dynamic_graph.operator_metatypes import OPERATOR_METATYPES
from nncf.hw_config import HWConfig
from nncf.dynamic_graph.input_wrapping import MODEL_INPUT_OP_NAME
from nncf.nncf_network import InsertionInfo, InsertionType, InsertionPointGraph, InsertionPointGraphNodeType, \
    InsertionPoint
from nncf.quantization.layers import QuantizerConfig, QuantizationMode
from nncf.utils import in_scope_list
from nncf.nncf_logger import logger as nncf_logger


class QuantizationTrait(Enum):
    """General, hardware-agnostic specifications for the relation of operators to quantization.
    Hardware-specific quantization configuration is handled elsewhere."""
    NON_QUANTIZABLE = -1
    QUANTIZATION_AGNOSTIC = 0
    INPUTS_QUANTIZABLE = 1


DEFAULT_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        Conv2dMetatype,
        Conv3dMetatype,
        ConvTranspose2dMetatype,
        ConvTranspose3dMetatype,
        DepthwiseConv2dSubtype,
        DepthwiseConv3dSubtype,
        LinearMetatype,
        HardTanhMetatype,
        TanhMetatype,
        ELUMetatype,
        PRELUMetatype,
        LayerNormMetatype,
        GELUMetatype,
        SigmoidMetatype,
        AddMetatype,
        MulMetatype,
        DivMetatype,
        ExpMetatype,
        ErfMetatype,
        MatMulMetatype,
        MeanMetatype,
        RoundMetatype,
        PixelShuffleMetatype,
        BatchNormMetatype
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        EmbeddingMetatype,
        SoftmaxMetatype
    ],
}  # type: Dict[QuantizationTrait, List[OperatorMetatype]]


class QuantizerInsertionInfo(InsertionInfo):
    def __init__(self, op_exec_context: OperationExecutionContext, is_input=False, is_output=False,
                 shape_to_operate_on=None,
                 quantizers_between_quantizable_layers: 'QuantizersBetweenQuantizableLayers' = None):
        super().__init__(op_exec_context, is_input, is_output, shape_to_operate_on)
        self.quantizers_between_quantizable_layers = quantizers_between_quantizable_layers

    @staticmethod
    def from_insertion_info(insertion_info: InsertionInfo) -> 'QuantizerInsertionInfo':
        result = QuantizerInsertionInfo(op_exec_context=insertion_info.op_exec_context,
                                        is_input=insertion_info.is_input,
                                        is_output=insertion_info.is_output,
                                        shape_to_operate_on=insertion_info.shape_to_operate_on)
        result.linked_op_exec_contexts = insertion_info.linked_op_exec_contexts
        return result

class PropagatingQuantizer:
    """Used in conjunction with QuantizerPropagationStateGraph to keep track of
       the allowed quantization configs corresponding to the model operation node
       whose inputs it quantizes, and also of the nodes/edges in the model control
       graph that this quantizer affects. It should be moved against the data flow of
       the model, tracking the affected nodes and edges of
       QuantizerPropagationStateGraph. No actual quantization modules are used here,
       only the associated configs (such as bitwidths, modes, signed/unsigned
       attributes etc.)"""

    def __init__(self, id_: int, quant_configs: List[QuantizerConfig], init_location_node_key: str,
                 unified_scale: bool = False):
        self._potential_quant_configs = quant_configs  # type: List[QuantizerConfig]
        self.affected_edges = set()
        self.affected_ip_nodes = set()
        self.propagation_path = []
        self.current_location_node_key = init_location_node_key
        self.last_accepting_location_node_key = None
        self.id = id_
        self.unified_scale = unified_scale

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def potential_quant_configs(self) -> List[QuantizerConfig]:
        return self._potential_quant_configs


class TransitionStatus(Enum):
    SHOULD_TRANSITION = 0
    SHOULD_MERGE = 1
    SHOULD_NOT_TRANSITION = 2


class PropagationStrategy(Enum):
    CONSERVATIVE = 0  # While propagating up through a downward-branching node,
                      # do not propagate if the propagation results in narrowing the list of
                      # quantization variants available to quantizers on neighbouring branches
    AGGRESSIVE = 1


class QuantizerPropagationStateGraphNodeType(Enum):
    INSERTION_POINT = 0
    OPERATOR = 1
    AUXILIARY_BARRIER = 2


class UnifiedScalePropagatingQuantizerGroupManager:
    def __init__(self):
        self._next_gid = 0
        self._group_vs_prop_quants_dict = {}  # type: Dict[int, Set[PropagatingQuantizer]]

    def _get_next_gid(self) -> int:
        retval = self._next_gid
        self._next_gid += 1
        return retval

    def register_group(self, prop_quants: Set[PropagatingQuantizer]) -> int:
        for pq in prop_quants:
            for gid, group in self._group_vs_prop_quants_dict.items():
                assert pq not in group, "Propagating quantizer #{} is already registered in a group {}!".format(pq.id,
                                                                                                                gid)
        gid = self._get_next_gid()
        self._group_vs_prop_quants_dict[gid] = prop_quants
        return gid

    def add_to_group(self, target_gid: int, prop_quant: PropagatingQuantizer):
        for gid, group in self._group_vs_prop_quants_dict.items():
            if target_gid != gid:
                assert prop_quant not in group, "Tried to add propagating quantizer #{} to group #{}, " \
                                                "but it is already registered in a group {}!".format(prop_quant.id,
                                                                                                     target_gid,
                                                                                                     gid)
        self._group_vs_prop_quants_dict[target_gid].add(prop_quant)

    def remove_from_group(self, group: int, prop_quant: PropagatingQuantizer):
        self._group_vs_prop_quants_dict[group].remove(prop_quant)

    def get_group_vs_prop_quants_dict(self) -> Dict[int, Set[PropagatingQuantizer]]:
        return copy(self._group_vs_prop_quants_dict)

    def get_group_id_by_propagating_quantizer_id(self, requested_pqid: int) -> Optional[int]:
        for gid, group in self._group_vs_prop_quants_dict.items():
            for pq in group:
                if pq.id == requested_pqid:
                    return gid
        return None

    def merge_groups(self, merge_to_gid: int, merge_from_gid: int):
        if merge_to_gid == merge_from_gid:
            return
        self._group_vs_prop_quants_dict[merge_to_gid].update(self._group_vs_prop_quants_dict[merge_from_gid])
        self._group_vs_prop_quants_dict.pop(merge_from_gid)


class QuantizerPropagationStateGraph(nx.DiGraph):
    """This class is based upon InsertionPointGraph and represents
       a"chessboard" for PropagatingQuantizer items.  It tracks the current state of
       quantizer propagation by associating the operator and insertion point nodes and
       edges to propagating quantizers, if any. It can move a propagating quantizer
       via own edges and mark its progress through the graph, which is required for
       resolving situations when multiple quantizers attempt to proceed via one and
       the same graph node/edge. This class is mainly operated upon by the
       QuantizerPropagationSolver objects."""
    PROPAGATING_QUANTIZER_NODE_ATTR = "propagating_quantizer"
    AFFECTING_PROPAGATING_QUANTIZERS_ATTR = "affecting_propagating_quantizers"
    QUANTIZATION_TRAIT_NODE_ATTR = "quantization_trait"
    ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR = "allowed_input_quantization_types"
    OPERATOR_METATYPE_NODE_ATTR = "op_meta"
    OPERATOR_SCOPE = "op_scope"
    INSERTION_POINT_DATA_NODE_ATTR = "insertion_point"
    NODE_TYPE_NODE_ATTR = "node_type"
    BARRIER_NODE_KEY_POSTFIX = "BARRIER"

    def __init__(self, ip_graph: InsertionPointGraph, ignored_scopes=None):
        super().__init__()
        ip_graph = deepcopy(ip_graph)
        self._created_prop_quantizer_counter = 0

        self._ignored_scopes = deepcopy(ignored_scopes)
        self.ignored_node_keys = []

        self._unified_scale_group_manager = UnifiedScalePropagatingQuantizerGroupManager()
        self._input_node_keys_vs_contexts = {}  # type: Dict[str, InputAgnosticOperationExecutionContext]

        barrier_node_extra_edges = []
        for node_key, node in ip_graph.nodes.items():
            qpg_node = {
                self.NODE_TYPE_NODE_ATTR: \
                    self.ipg_node_type_to_qpsg_node_type(node[InsertionPointGraph.NODE_TYPE_NODE_ATTR])}
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.INSERTION_POINT:
                qpg_node[self.PROPAGATING_QUANTIZER_NODE_ATTR] = None
                qpg_node[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
                qpg_node[self.INSERTION_POINT_DATA_NODE_ATTR] = node[
                    InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            elif node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                qpg_node[self.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR] = set()
                qpg_node[
                    self.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.NON_QUANTIZABLE
                qpg_node[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
                qpg_node[self.OPERATOR_METATYPE_NODE_ATTR] = node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR]
                node_ia_op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic  # type: InputAgnosticOperationExecutionContext
                qpg_node[self.OPERATOR_SCOPE] = node_ia_op_exec_context.scope_in_model
                node_scope = str(node_ia_op_exec_context)

                op_name = node_ia_op_exec_context.operator_name
                if op_name == MODEL_INPUT_OP_NAME:
                    self._input_node_keys_vs_contexts[node_key] = node_ia_op_exec_context

                if in_scope_list(node_scope, self._ignored_scopes):
                    self.ignored_node_keys.append(node_key)
                    qpg_node_barrier = {
                        self.NODE_TYPE_NODE_ATTR: QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER,
                        'label': QuantizerPropagationStateGraph.BARRIER_NODE_KEY_POSTFIX}
                    barrier_node_key = self.get_barrier_node_key(node_key)
                    self.add_node(barrier_node_key, **qpg_node_barrier)
                    barrier_node_extra_edges.append((barrier_node_key, node_key))

            self.add_node(node_key, **qpg_node)

        for from_node, to_node, edge_data in ip_graph.edges(data=True):
            edge_data[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
            self.add_edge(from_node, to_node, **edge_data)

        for u_node_key, v_node_key in barrier_node_extra_edges:
            edge_attr = {QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR: []}
            next_v_node_key = list(self.succ[v_node_key].keys())[0]  # POST HOOK v
            self.add_edge(v_node_key, u_node_key, **edge_attr)
            self.add_edge(u_node_key, next_v_node_key, **edge_attr)
            self.remove_edge(v_node_key, next_v_node_key)

    @staticmethod
    def ipg_node_type_to_qpsg_node_type(ipg_node_type: InsertionPointGraphNodeType) \
        -> QuantizerPropagationStateGraphNodeType:
        if ipg_node_type == InsertionPointGraphNodeType.INSERTION_POINT:
            return QuantizerPropagationStateGraphNodeType.INSERTION_POINT
        if ipg_node_type == InsertionPointGraphNodeType.OPERATOR:
            return QuantizerPropagationStateGraphNodeType.OPERATOR
        raise RuntimeError("Invalid insertion point graph node type.")

    @staticmethod
    def get_barrier_node_key(node_key: str):
        return QuantizerPropagationStateGraph.BARRIER_NODE_KEY_POSTFIX + node_key

    # pylint:disable=too-many-branches
    def merge_quantizer_into_path(self, prop_quantizer: PropagatingQuantizer, path: List):
        curr_node = self.nodes[prop_quantizer.current_location_node_key]
        curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        surviving_quantizers = []  # type: List[PropagatingQuantizer]
        for from_node_key, to_node_key in path:
            edge = self.edges[from_node_key, to_node_key]
            potential_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if potential_quantizers:
                surviving_quantizers = potential_quantizers
                break
            from_node = self.nodes[from_node_key]
            potential_quantizer = from_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
            if potential_quantizer is None:
                if from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]:
                    potential_quantizer = \
                        from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR][0]

            if potential_quantizer is not None:
                prop_quantizer.affected_edges.add((from_node_key, to_node_key))
                edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
                surviving_quantizers.append(potential_quantizer)
                break

        if surviving_quantizers:
            for pq in surviving_quantizers:
                pq.affected_edges.update(prop_quantizer.affected_edges)
                for from_node_key, to_node_key in prop_quantizer.affected_edges:
                    from_node = self.nodes[from_node_key]
                    from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                    if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                        # pylint:disable=line-too-long
                        self.nodes[from_node_key][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(pq)
            if prop_quantizer.unified_scale:
                gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
                for other_pq in surviving_quantizers:
                    if other_pq.unified_scale:
                        other_gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(
                            other_pq.id)
                        self._unified_scale_group_manager.merge_groups(gid, other_gid)
                    else:
                        self._unified_scale_group_manager.add_to_group(gid, other_pq)

            for affected_edge_tuple in prop_quantizer.affected_edges:
                edge = self.edges[affected_edge_tuple]
                affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                for pq in surviving_quantizers:
                    affecting_quantizers.append(pq)
            self.remove_propagating_quantizer(prop_quantizer)
        else:
            raise RuntimeError("Surviving_quantizers not found !"
                               " Nodes quantized with quantizer #{} will be lost".format(prop_quantizer.id))

    def backtrack_propagation_until_accepting_location(self, prop_quantizer: PropagatingQuantizer) -> Optional[
            PropagatingQuantizer]:
        if prop_quantizer.last_accepting_location_node_key is None:
            # The quantizer was stillborn.
            # If there are quantizer-affected inbound edges, should transfer this quantizer's
            # affected edges and nodes to the inbound edge quantizers
            curr_node_key = prop_quantizer.current_location_node_key
            inbound_affecting_quantizers = set()
            for in_edge_key in self.in_edges(curr_node_key):
                in_edge = self.edges[in_edge_key]
                inbound_affecting_quantizers.update(
                    in_edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR])

            for inbound_pq in inbound_affecting_quantizers:
                inbound_pq.affected_edges.update(prop_quantizer.affected_edges)
                inbound_pq.affected_ip_nodes.update(prop_quantizer.affected_ip_nodes)
            for edge in prop_quantizer.affected_edges:
                self.edges[edge][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] += list(
                    inbound_affecting_quantizers)
            for ip_node_key in prop_quantizer.affected_ip_nodes:
                self.nodes[ip_node_key][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] += list(
                    inbound_affecting_quantizers)

            self.remove_propagating_quantizer(prop_quantizer)
            return None

        curr_node_key = prop_quantizer.current_location_node_key
        curr_node = self.nodes[curr_node_key]
        curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        while prop_quantizer.current_location_node_key != prop_quantizer.last_accepting_location_node_key:
            from_node_key, to_node_key = prop_quantizer.propagation_path.pop()

            edge = self.edges[from_node_key, to_node_key]
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].remove(prop_quantizer)
            prop_quantizer.affected_edges.remove((from_node_key, to_node_key))
            from_node = self.nodes[from_node_key]
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].remove(prop_quantizer)
                prop_quantizer.affected_ip_nodes.remove(from_node_key)

            to_node = self.nodes[to_node_key]
            to_node_type = to_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if to_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                prop_quantizer.current_location_node_key = to_node_key

        target_ip_node_key = prop_quantizer.current_location_node_key
        target_node = self.nodes[target_ip_node_key]
        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        return prop_quantizer

    def add_propagating_quantizer(self, qconf_list: List[QuantizerConfig], ip_node_key: str,
                                  unified_scale: bool = False) -> PropagatingQuantizer:
        prop_quantizer = PropagatingQuantizer(self._get_next_prop_quantizer_id(), qconf_list, ip_node_key,
                                              unified_scale)

        if unified_scale:
            self._unified_scale_group_manager.register_group({prop_quantizer})

        ip_node = self.nodes[ip_node_key]
        ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        ip_type = ip_node[QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR].insertion_type

        if ip_type != InsertionType.OPERATOR_PRE_HOOK:
            # The insertion point key should immediately precede a quantizable op,
            # otherwise it is hard to determine affected node here (although possible)
            raise RuntimeError("Can only add propagating quantizers into pre-hook spots!")

        affected_op_node_key = next(self.successors(ip_node_key))
        affected_op_node = self.nodes[affected_op_node_key]

        affected_op_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        initial_edge_key = (ip_node_key, affected_op_node_key)
        initial_edge = self.edges[initial_edge_key]
        initial_edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
        prop_quantizer.affected_edges.add(initial_edge_key)
        prop_quantizer.affected_ip_nodes.add(ip_node_key)
        return prop_quantizer

    def clone_propagating_quantizer(self, prop_quantizer: PropagatingQuantizer) -> PropagatingQuantizer:
        cloned_prop_quant = deepcopy(prop_quantizer)
        cloned_prop_quant.id = self._get_next_prop_quantizer_id()
        for edge_tuple in cloned_prop_quant.affected_edges:
            edge = self.edges[edge_tuple]
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(cloned_prop_quant)
        for node_key in cloned_prop_quant.affected_ip_nodes:
            node = self.nodes[node_key]
            node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(cloned_prop_quant)

        if cloned_prop_quant.unified_scale:
            gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
            self._unified_scale_group_manager.add_to_group(gid, cloned_prop_quant)

        return cloned_prop_quant

    def remove_propagating_quantizer(self, prop_quantizer: PropagatingQuantizer):
        for edge_tuple in prop_quantizer.affected_edges:
            edge = self.edges[edge_tuple]
            affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)
        for node_key in prop_quantizer.affected_ip_nodes:
            node = self.nodes[node_key]
            affecting_quantizers = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)

        node_key = prop_quantizer.current_location_node_key
        self.nodes[node_key][QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        prop_quantizer.affected_ip_nodes.clear()
        prop_quantizer.affected_edges.clear()
        if prop_quantizer.unified_scale:
            gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
            self._unified_scale_group_manager.remove_from_group(gid, prop_quantizer)

    def propagate_quantizer_via_path(self, prop_quantizer: PropagatingQuantizer, path: List) -> PropagatingQuantizer:
        curr_node_key = prop_quantizer.current_location_node_key
        curr_node = self.nodes[curr_node_key]
        existing_quantizer = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
        if existing_quantizer is not None and existing_quantizer.id == prop_quantizer.id:
            curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        for edge_tuple in path:
            edge = self.edges[edge_tuple]
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
            prop_quantizer.affected_edges.add(edge_tuple)
            prop_quantizer.propagation_path.append(edge_tuple)
            from_node_key = edge_tuple[0]
            from_node = self.nodes[from_node_key]
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
                prop_quantizer.affected_ip_nodes.add(from_node_key)
                if self._is_position_accepting(from_node_key):
                    prop_quantizer.last_accepting_location_node_key = from_node_key

        target_ip_node_key = path[-1][0]
        prop_quantizer.current_location_node_key = target_ip_node_key
        target_node = self.nodes[target_ip_node_key]
        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        return prop_quantizer

    def get_quantizable_op_nodes_immediately_dominated_by_node(self, node_key) -> List[str]:
        ret_node_key_list = []

        def recursive_helper(curr_node_key: str, target_node_list: List[str]):
            successors = self.successors(curr_node_key)
            for successor_key in successors:
                successor = self.nodes[successor_key]
                successor_node_type = successor[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                if successor_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                    trait = successor[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                    if not trait == QuantizationTrait.QUANTIZATION_AGNOSTIC:
                        target_node_list.append(successor_key)
                        return
                recursive_helper(successor_key, target_node_list)

        recursive_helper(node_key, ret_node_key_list)
        return ret_node_key_list

    def get_paths_to_immediately_dominating_insertion_points(self, insertion_point_node_key: str) -> List[List]:
        """Paths are lists of edges."""
        paths = []

        def recursive_helper(curr_edge, curr_path, all_paths):
            curr_path.append(curr_edge)
            curr_node_key = curr_edge[0]
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                all_paths.append(curr_path)
                return

            for in_edge in self.in_edges(curr_node_key):
                path_copy = deepcopy(curr_path)
                recursive_helper(in_edge, path_copy, all_paths)

        for in_edge in self.in_edges(insertion_point_node_key):
            if self.nodes[in_edge[0]][QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR] == \
                    QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER:
                return paths
            recursive_helper(in_edge, [], paths)
        return paths

    def get_visualized_graph(self):
        out_graph = nx.DiGraph()
        unified_scale_group_vs_pq_node_id_dict = {}  # type: Dict[int, List[str]]
        for node_key, node in self.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[
                    QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                label = "IP: {}".format(insertion_point_data.insertion_type)
                out_graph.add_node(node_key, label=label, color="red")
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    prop_quantizer = node[
                        QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]  # type: PropagatingQuantizer
                    quant_node_key = "Quantizer #{}".format(prop_quantizer.id)
                    if prop_quantizer.potential_quant_configs:
                        quant_configs_str_list = [str(conf) for conf in prop_quantizer.potential_quant_configs]
                    else:
                        quant_configs_str_list = ["!!! NONE !!!]"]
                    sub_label = '[' + ',\n'.join(quant_configs_str_list) + ']'
                    quant_node_label = quant_node_key + '\n' + "T: {}\n".format(sub_label)
                    out_graph.add_node(quant_node_key,
                                       color="blue", label=quant_node_label)
                    out_graph.add_edge(quant_node_key, node_key,
                                       style="dashed")
                    if prop_quantizer.unified_scale:
                        gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(
                            prop_quantizer.id)
                        if gid in unified_scale_group_vs_pq_node_id_dict:
                            unified_scale_group_vs_pq_node_id_dict[gid].append(quant_node_key)
                        else:
                            unified_scale_group_vs_pq_node_id_dict[gid] = [quant_node_key]

            elif node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                out_graph.add_node(node_key)
            elif node_type == QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER:
                out_graph.add_node(node_key, color='green', label=node['label'])
            else:
                raise RuntimeError("Invalid QuantizerPropagationStateGraph node!")
        for u, v in self.edges:
            edge = self.edges[u, v]
            attrs = {}
            affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if affecting_quantizers:
                label = ", ".join([str(pq.id) for pq in affecting_quantizers])
                attrs = {"color": "blue", "label": label}
            out_graph.add_edge(u, v, **attrs)

        for gid, group_pq_node_keys in unified_scale_group_vs_pq_node_id_dict.items():
            if len(group_pq_node_keys) < 2:
                continue
            curr_elt_iter = iter(group_pq_node_keys)
            next_elt_iter = iter(group_pq_node_keys)
            _ = next(next_elt_iter)  # in order not to use more_itertools.consume
            done = False
            while not done:
                curr_pq_node_key = next(curr_elt_iter)
                try:
                    next_pq_node_key = next(next_elt_iter)
                except StopIteration:
                    done = True
                    next_pq_node_key = group_pq_node_keys[0]  # back to the first elt
                out_graph.add_edge(curr_pq_node_key, next_pq_node_key, arrowhead="none",
                                   style="dotted",
                                   label="Unified group {}".format(gid))

        return out_graph

    def traverse_graph(self, curr_node_key: str,
                       traverse_function: Callable[[str, Any], Tuple[bool, Any]],
                       output: Any,
                       traverse_forward: bool = True) -> Any:
        return self._traverse_graph_recursive_helper(curr_node_key, traverse_function, output, traverse_forward)

    def _traverse_graph_recursive_helper(self, curr_node_key: str,
                                         traverse_function: Callable[[str, Any], Tuple[bool, Any]],
                                         output: Any, traverse_forward: bool):
        is_finished, output = traverse_function(curr_node_key, output)
        node_keys_holder = self.succ if traverse_forward else self.pred
        if not is_finished:
            for node_key in node_keys_holder[curr_node_key]:
                self._traverse_graph_recursive_helper(node_key, traverse_function, output, traverse_forward)
        return output

    def _get_next_prop_quantizer_id(self):
        self._created_prop_quantizer_counter += 1
        return self._created_prop_quantizer_counter

    def _is_position_accepting(self, ip_node_key: str):
        node = self.nodes[ip_node_key]
        insertion_type = node[QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR].insertion_type
        if insertion_type == InsertionType.OPERATOR_POST_HOOK:
            return True
        return False

    def get_unified_scale_group_id_by_propagating_quantizer_id(self, pqid: int) -> int:
        return self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(pqid)

    def get_input_quantizer_ids(self) -> Dict[InputAgnosticOperationExecutionContext, List[int]]:
        retval = {}  # type: Dict[InputAgnosticOperationExecutionContext, List[int]]

        def recursive_helper(curr_node_key: str, curr_input_quantizer_ids_list: List[int]):
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]

            if curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                pq = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if pq is not None:
                    curr_input_quantizer_ids_list.append(pq.id)
                    return
            elif curr_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = curr_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if not trait == QuantizationTrait.QUANTIZATION_AGNOSTIC:
                    return
            elif curr_node_type == QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER:
                return

            for successor_key in self.successors(curr_node_key):
                recursive_helper(successor_key, curr_input_quantizer_ids_list)

        for input_node_key, input_node_context in self._input_node_keys_vs_contexts.items():
            current_input_quantizer_ids = []
            recursive_helper(input_node_key, current_input_quantizer_ids)
            retval[input_node_context] = current_input_quantizer_ids

        return retval


class QuantizersBetweenQuantizableLayers:
    """ Contains locations of quantizers between inputs quantizable layers: input agnostic operation execution context
    for activations and scope - for quantized modules """

    def __init__(self):
        self.activation_quantizer_ctxs = set()  # type: Set[InputAgnosticOperationExecutionContext]
        self.quantized_module_scopes = set()  # type: Set['Scope']

    def add_activation_quantizer_ctx(self, iap_ctx: InputAgnosticOperationExecutionContext):
        self.activation_quantizer_ctxs.add(iap_ctx)

    def add_quantized_module_scope(self, scope: 'Scope'):
        self.quantized_module_scopes.add(scope)

    def __bool__(self) -> bool:
        return bool(self.activation_quantizer_ctxs) and bool(self.quantized_module_scopes)


class QuantizerPropagationSolver:
    """Analyzes a fresh QuantizerPropagationStateGraph object according to HW
       configuration supplied in the initializer and produces the list of insertion
       commands that correspond to the final state of the quantizer propagation graph
       when the model has the most contol flow graph edges quantized according to HW
       capabilities."""

    DEFAULT_QUANTIZATION_TYPES = [QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        per_channel=False)]

    def __init__(self, ignored_scopes=None, hw_config: HWConfig = None,
                 debug_interface: 'QuantizationDebugInterface' = None,
                 propagation_strategy: PropagationStrategy = PropagationStrategy.AGGRESSIVE,
                 default_qconfig_list: List[QuantizerConfig] = None,
                 input_infos: List[ModelInputInfo] = None):
        self.quantizers_between_quantizable_layers_per_key = {}  # type: Dict[str, QuantizersBetweenQuantizableLayers]
        self.default_qlobal_qconfig_list = default_qconfig_list
        self._hw_config = hw_config  # type: HWConfig
        self._debug_interface = debug_interface
        self._propagation_strategy = propagation_strategy  # TODO: determine from config
        self._operator_quantization_trait_map = self.get_operator_quantization_traits_map()
        self._operator_allowed_qconfigs_map = self._get_operator_qconfigs_map()
        self._input_infos = input_infos

        if self._hw_config is not None:
            self._unified_scales_operation_set = self._hw_config.get_operations_with_unified_scales()
        else:
            self._unified_scales_operation_set = {}

        # Will handle the "wildcard" quantization situation for the time being
        if default_qconfig_list is not None:
            for op_meta, qconf_list in self._operator_allowed_qconfigs_map.items():
                trait = self._operator_quantization_trait_map[op_meta]
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    # !!! FIXME !!! Ensure that INPUTS_QUANTIZABLE ops always have non-None qconf_list
                    if HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconf_list):
                        self._operator_allowed_qconfigs_map[op_meta] = default_qconfig_list
        self._active_propagating_quantizers_queue = deque()
        self._finished_propagating_quantizers = []  # type: List[PropagatingQuantizer]
        self._ignored_scopes = ignored_scopes

        self._potential_quantizers = {}

    def run_on_ip_graph(self, ip_graph: InsertionPointGraph) -> Dict[QuantizerInsertionInfo,
                                                                     Optional[List[QuantizerConfig]]]:
        """ The main function to be used on an InsertionPointGraph to produce
            the list of insertion commands and configs corresponding to the final quantized
            graph state."""
        quant_prop_graph = QuantizerPropagationStateGraph(ip_graph, self._ignored_scopes)
        quant_prop_graph = self.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
        quant_prop_graph = self.setup_initial_quantizers(quant_prop_graph)
        iteration_counter = 0
        while self._active_propagating_quantizers_queue:
            prop_quantizer = self._active_propagating_quantizers_queue.pop()
            if self._debug_interface is not None:
                self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, str(iteration_counter))
            quant_prop_graph = self.propagation_step(prop_quantizer, quant_prop_graph)
            iteration_counter += 1

        if self._input_infos is not None:
            self._filter_integer_input_quantizers(quant_prop_graph)

        if self._debug_interface is not None:
            self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, "final")

        retval = {}

        non_unified_final_prop_quantizers = set()  # type: Set[PropagatingQuantizer]
        unified_final_prop_quantizers = {}  # type: Dict[int, Set[PropagatingQuantizer]]

        self.quantizers_between_quantizable_layers_per_key = \
            self._get_quantizers_between_quantizable_layers_per_node_key(
                quant_prop_graph, self._finished_propagating_quantizers)

        for finished_prop_quantizer in self._finished_propagating_quantizers:
            if finished_prop_quantizer.unified_scale:
                # Handle unified scale quantizers separately since they require special InsertionInfo construction
                gid = quant_prop_graph.get_unified_scale_group_id_by_propagating_quantizer_id(
                    finished_prop_quantizer.id)
                if gid not in unified_final_prop_quantizers:
                    unified_final_prop_quantizers[gid] = set()
                unified_final_prop_quantizers[gid].add(finished_prop_quantizer)
            else:
                non_unified_final_prop_quantizers.add(finished_prop_quantizer)

        for non_unified_prop_quant in non_unified_final_prop_quantizers:
            insertion_info = self._get_insertion_info_for_propagating_quantizer(non_unified_prop_quant,
                                                                                quant_prop_graph)
            retval[insertion_info] = non_unified_prop_quant.potential_quant_configs

        for unified_prop_quantizer_set in unified_final_prop_quantizers.values():
            insertion_infos_vs_qconfigs = {}  # type: Dict[QuantizerInsertionInfo, List[QuantizerConfig]]
            for prop_quant in unified_prop_quantizer_set:
                insertion_info = self._get_insertion_info_for_propagating_quantizer(prop_quant, quant_prop_graph)
                insertion_infos_vs_qconfigs[insertion_info] = prop_quant.potential_quant_configs

            # The primary insertion point (to be associated with the actual quantizer module, not just hooks to it)
            # will be determined based on the string representation of said insertion point, to avoid random selection
            all_insertion_infos = sorted(list(insertion_infos_vs_qconfigs.keys()),
                                         key=lambda x: str(x.op_exec_context.input_agnostic))
            primary_insertion_info = all_insertion_infos[0]
            linked_insertion_infos = all_insertion_infos[1:]
            primary_insertion_info.linked_op_exec_contexts = [x.op_exec_context for x in linked_insertion_infos]
            retval[primary_insertion_info] = insertion_infos_vs_qconfigs[primary_insertion_info]

        return retval

    def _get_insertion_info_for_propagating_quantizer(self, prop_quant: PropagatingQuantizer,
                                                      quant_prop_graph: QuantizerPropagationStateGraph) -> \
            QuantizerInsertionInfo:
        final_node_key = prop_quant.current_location_node_key
        final_node = quant_prop_graph.nodes[final_node_key]
        insertion_point = final_node[
            QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
        # TODO: fix this, rethink InsertionInfo here and elsewhere
        insertion_info = QuantizerInsertionInfo(OperationExecutionContext(
            operator_name=insertion_point.ia_op_exec_context.operator_name,
            scope_in_model=insertion_point.ia_op_exec_context.scope_in_model,
            call_order=insertion_point.ia_op_exec_context.call_order,
            tensor_metas=[None],
        ), quantizers_between_quantizable_layers=self.quantizers_between_quantizable_layers_per_key[final_node_key])
        return insertion_info

    def propagation_step(self, curr_prop_quantizer: PropagatingQuantizer,
                         quant_prop_graph: QuantizerPropagationStateGraph) -> QuantizerPropagationStateGraph:
        """Returns an updated curr_prop_quantizer state if the quantizer is not
           yet in its final (accepting) position, and None if the quantizer is in its
           final location.  The location before and after the step should correspond to
           some insertion point."""
        # TODO: full-fledged discrete finite automata approach? Switch to traversing a graph
        # consisting of insertion points only, with reversed edges holding associated operator data?
        curr_node_key = curr_prop_quantizer.current_location_node_key
        curr_node = quant_prop_graph.nodes[curr_prop_quantizer.current_location_node_key]
        curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
        assert curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT

        # Assumption: paths are at most 2 edges - either one edge to neighbouring insertion point
        # or one edge to operation and next edge to its own neighbouring insertion point.

        paths = quant_prop_graph.get_paths_to_immediately_dominating_insertion_points(curr_node_key)
        if not paths:
            prop_quantizer = quant_prop_graph.backtrack_propagation_until_accepting_location(curr_prop_quantizer)
            if prop_quantizer is not None:
                self._finished_propagating_quantizers.append(prop_quantizer)
            return quant_prop_graph

        surviving_prop_quantizers = []

        prop_quantizers_to_process = [curr_prop_quantizer]
        for _ in range(1, len(paths)):
            additional_prop_quantizer = quant_prop_graph.clone_propagating_quantizer(curr_prop_quantizer)
            prop_quantizers_to_process.append(additional_prop_quantizer)

        pqs_and_paths = zip(paths, prop_quantizers_to_process)
        for path, prop_quantizer in pqs_and_paths:
            status = self.check_transition_via_path(prop_quantizer, path, quant_prop_graph)
            if status == TransitionStatus.SHOULD_NOT_TRANSITION:
                prop_quantizer = quant_prop_graph.backtrack_propagation_until_accepting_location(prop_quantizer)
                if prop_quantizer is not None:
                    self._finished_propagating_quantizers.append(prop_quantizer)
            elif status == TransitionStatus.SHOULD_TRANSITION:
                prop_quantizer = quant_prop_graph.propagate_quantizer_via_path(prop_quantizer, path)
                surviving_prop_quantizers.append(prop_quantizer)
            elif status == TransitionStatus.SHOULD_MERGE:
                # The surviving quantizer will have its "affected edges" set extended
                # by the corresponding set of the merged quantizer. The assumption
                # here is that the surviving quantizer should never have to cross
                # such a "merge point" while backtracking to an accepting location.

                quant_prop_graph.merge_quantizer_into_path(prop_quantizer, path)

        for prop_quantizer in surviving_prop_quantizers:
            self._active_propagating_quantizers_queue.appendleft(prop_quantizer)
        return quant_prop_graph

    def get_allowed_quantizer_configs_for_operator(self, quant_det_id: OperatorMetatype) -> List[QuantizerConfig]:
        return self._operator_allowed_qconfigs_map[quant_det_id]

    def set_allowed_quantization_types_for_operator_nodes(self, quant_prop_graph: QuantizerPropagationStateGraph):
        for node_key, node in quant_prop_graph.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                quant_det_id = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                if quant_det_id is None:
                    warnings.warn("Unknown metatype for operator node: {}".format(node_key))
                    trait = QuantizationTrait.QUANTIZATION_AGNOSTIC
                else:
                    trait = self._operator_quantization_trait_map[quant_det_id]
                node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] = trait
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    node[QuantizerPropagationStateGraph.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR] = \
                        self.get_allowed_quantizer_configs_for_operator(quant_det_id)
        return quant_prop_graph

    def get_operator_quantization_traits_map(self) -> Dict[OperatorMetatype, QuantizationTrait]:
        # TODO: ensure that there are no name collisions between ops in different torch subpackages with the same name
        retval = {}
        if self._hw_config is None:
            for op_meta in OPERATOR_METATYPES.registry_dict.values():
                retval[op_meta] = QuantizationTrait.QUANTIZATION_AGNOSTIC  # Default value
            for trait, meta_list in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
                for op_meta in meta_list:  # type: OperatorMetatype
                    retval[op_meta] = trait
        else:
            op_meta_vs_qconfs_map = self._hw_config.get_metatype_vs_quantizer_configs_map()
            for op_meta, qconf_list in op_meta_vs_qconfs_map.items():
                if HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconf_list):
                    trait = self._get_trait_for_op_meta_not_specified_in_hw_config(op_meta)
                elif HWConfig.is_wildcard_quantization(qconf_list):
                    for default_trait, meta_list in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
                        if op_meta in meta_list:
                            trait = default_trait
                            break
                    else:
                        trait = QuantizationTrait.QUANTIZATION_AGNOSTIC
                else:
                    trait = QuantizationTrait.INPUTS_QUANTIZABLE
                retval[op_meta] = trait
        return retval

    @staticmethod
    def _get_trait_for_op_meta_not_specified_in_hw_config(op_meta: OperatorMetatype) -> QuantizationTrait:
        if not op_meta.hw_config_names:
            # The metatype might not have an associated name in the config
            # namespace (yet) - use default trait
            for default_trait, meta_list in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
                if op_meta in meta_list:
                    trait = default_trait
                    break
            else:
                trait = QuantizationTrait.QUANTIZATION_AGNOSTIC
                # TODO: think of switching to this?
                # raise RuntimeError("Operation metatype {} encountered, but it has no default "
                #                    "quantization trait and the HW config entry is not given for it - "
                #                    "cannot determine how to quantize it!".format(op_meta))
        else:
            # There IS a valid HW config name for the metatype, but it is deliberately not specified
            # in the config, which means that it should execute in FP32
            trait = QuantizationTrait.NON_QUANTIZABLE

        return trait

    def _get_operator_qconfigs_map(self) -> Dict[OperatorMetatype, List[QuantizerConfig]]:
        # TODO: ensure that there are no name collisions between ops in different torch subpackages with the same name
        retval = {}
        if self._hw_config is None:
            for op_meta in OPERATOR_METATYPES.registry_dict.values():
                retval[op_meta] = []  # Default value, corresponds to wildcard quantization
            for trait, meta_list in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    for op_meta in meta_list:  # type: OperatorMetatype
                        if self.default_qlobal_qconfig_list is not None:
                            retval[op_meta] = deepcopy(self.default_qlobal_qconfig_list)
                        else:
                            retval[op_meta] = deepcopy(self.DEFAULT_QUANTIZATION_TYPES)
                elif trait == QuantizationTrait.NON_QUANTIZABLE:
                    for op_meta in meta_list:  # type: OperatorMetatype
                        retval[op_meta] = None
        else:
            retval = self._hw_config.get_metatype_vs_quantizer_configs_map()
        return retval

    def debug_visualize(self, quant_prop_graph: QuantizerPropagationStateGraph, dump_path: str):
        out_graph = quant_prop_graph.get_visualized_graph()
        active_ids_str = ", ".join([str(pq.id) for pq in self._active_propagating_quantizers_queue])
        finished_ids_str = ", ".join([str(pq.id) for pq in self._finished_propagating_quantizers])
        out_graph.graph['graph'] = {"label": "Propagating quantizers: {}\n" \
                                             "Finished quantizers: {}".format(active_ids_str, finished_ids_str),
                                    "labelloc": "t"}
        nx.drawing.nx_pydot.write_dot(out_graph, dump_path)

    def setup_initial_quantizers(self,
                                 quant_prop_graph: QuantizerPropagationStateGraph) -> QuantizerPropagationStateGraph:
        """Determines the initial subset of the nodes that must be quantized
           and corresponding allowed quantization configs (possibly multiple) for each
           quantizer."""
        for node_key, node in quant_prop_graph.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                if node_key in quant_prop_graph.ignored_node_keys:
                    continue

                preds = list(quant_prop_graph.predecessors(node_key))
                if not preds:
                    continue  # TODO: remove this once module insertion points are included in the IP graph
                # Should be immediately preceded by an insertion point.
                pred_ip_key = preds[0]
                pred_node = quant_prop_graph.nodes[pred_ip_key]
                pred_node_type = pred_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                assert pred_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT, \
                    "Invalid insertion point graph supplied for quantizer propagation!"

                if node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] in [
                        QuantizationTrait.NON_QUANTIZABLE,
                        QuantizationTrait.QUANTIZATION_AGNOSTIC]:
                    continue

                quant_det_id = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                qconf_list = self.get_allowed_quantizer_configs_for_operator(quant_det_id)

                # No need to place quantizers for FP32-forced ops, naturally
                assert qconf_list is not None

                is_unified_scale = quant_det_id in self._unified_scales_operation_set
                if is_unified_scale:
                    # Filtering out the per-channel cases in the unified scale scenario.
                    # In order to support unified per-channel scales, we will need to handle a situation
                    # when a per-channel shared (unified) quantizer on one branch passes a shape-chaning
                    # operation such as `view` or `transpose`, and the linked unified quantizer does not do
                    # so due to one or the other reason; the per-channel scale shapes to be applied therefore
                    # will be different.
                    # TODO: What possibly needs to be done in this direction:
                    # 1. keep ForwardTraceOnly ops in graph after all, to be able to track shape changes
                    # 2. transpose input tensors to the quantization modules on the fly to accomodate scale,
                    #    or vice versa, transpose scale to accomodate shape; need to handle exporting as well
                    per_tensor_qconf_list = list(filter(lambda x: x.per_channel is False, qconf_list))
                    op_meta_name = quant_det_id.__class__.__name__
                    if len(per_tensor_qconf_list) != len(qconf_list):
                        if not per_tensor_qconf_list:
                            raise RuntimeError(
                                "Unified scales currently do not support per-channel configuration - dropping"
                                "per-channel configuration options for {} resulted in no valid quantization "
                                "configs!".format(op_meta_name))
                        nncf_logger.warning(
                            "Unified scales currently do not support per-channel configuration - dropping"
                            "per-channel configuration options for {}".format(op_meta_name))
                        qconf_list = per_tensor_qconf_list

                prop_quantizer = quant_prop_graph.add_propagating_quantizer(qconf_list, pred_ip_key, is_unified_scale)
                self._active_propagating_quantizers_queue.appendleft(prop_quantizer)

        return quant_prop_graph

    # pylint:disable=too-many-return-statements
    def check_branching_transition(self, quant_prop_graph: QuantizerPropagationStateGraph,
                                   prop_quantizer: PropagatingQuantizer,
                                   branching_node_key: str) -> Optional[TransitionStatus]:
        """If a propagating quantizer advances through a node that branches
           downwards, the branches neighbouring to the one that the propagating quantizer
           had just propagated from will have the precision of the quantizer imposed upon
           them.  This is not always desirable - we might want to keep some branches in
           higher precision than the others. For this reason, this function checks whether
           the quantizer may safely advance through a branching node based on the possible
           configs of the quantizers it might affect by doing so."""
        dom_op_node_keys = quant_prop_graph.get_quantizable_op_nodes_immediately_dominated_by_node(
            branching_node_key)
        primary_possible_qconfigs = prop_quantizer.potential_quant_configs
        secondary_possible_qconfigs_dict = {}
        for op_node_key in dom_op_node_keys:
            op_node = quant_prop_graph.nodes[op_node_key]
            affecting_prop_quantizers = op_node[
                QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if not affecting_prop_quantizers:
                # The branch op is forced to be FP32 - should not proceed through the branch node.
                return TransitionStatus.SHOULD_NOT_TRANSITION
            secondary_possible_qconfigs = affecting_prop_quantizers[0].potential_quant_configs
            secondary_possible_qconfigs_dict[op_node_key] = secondary_possible_qconfigs
        primary_merged_qconfigs, \
        secondary_merged_qconfigs_dict = self.get_merged_qconfigs(primary_possible_qconfigs,
                                                                  secondary_possible_qconfigs_dict)
        if not primary_merged_qconfigs:
            # This quantizer's precision does not encompass the precisions of quantizers
            # propagating through downward branches.
            return TransitionStatus.SHOULD_NOT_TRANSITION

        if self._propagation_strategy == PropagationStrategy.CONSERVATIVE:
            for op_node_key, secondary_merged_qconfigs_list in secondary_merged_qconfigs_dict.items():
                if len(secondary_possible_qconfigs_dict[op_node_key]) != len(secondary_merged_qconfigs_list):
                    return TransitionStatus.SHOULD_NOT_TRANSITION

        return None

    def check_transition_via_path(self, prop_quantizer: PropagatingQuantizer, path: List,
                                  quant_prop_graph: QuantizerPropagationStateGraph) -> TransitionStatus:
        """Determines which action should be taken regarding the
           prop_quantizer's propagation via path, which may be one of many possible
           propagation paths."""
        for from_node_key, to_node_key in path:
            from_node = quant_prop_graph.nodes[from_node_key]
            if len(list(quant_prop_graph.successors(from_node_key))) > 1:
                # If a quantizer simply passes up through a downward-branching node, it may spoil the
                # precision for operations on neighbouring branches. Consider a 4-bit quantizer rising
                # through a branch node and an 8-bit quantizer arriving at the same node later. Therefore,
                # prior to allowing the quantizer to pass through a branching node we need to ensure that
                # the precision of the quantizer is a superset of precisions of the first non-quantization agnostic
                # operations on each branch.
                status = self.check_branching_transition(quant_prop_graph,
                                                         prop_quantizer,
                                                         from_node_key)
                if status is not None:
                    return status

            # Check if current edge to traverse is affected by any of the quantizers
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = from_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait in [QuantizationTrait.NON_QUANTIZABLE,
                             QuantizationTrait.INPUTS_QUANTIZABLE]:
                    return TransitionStatus.SHOULD_NOT_TRANSITION
            edge = quant_prop_graph.edges[from_node_key, to_node_key]
            potential_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if potential_quantizers:
                # Assuming that multiple affecting quantizers should all have the same quantization config
                # by construction
                curr_pq_configs = prop_quantizer.potential_quant_configs
                target_pq_configs = potential_quantizers[0].potential_quant_configs
                if curr_pq_configs == target_pq_configs or \
                        HWConfig.is_wildcard_quantization(curr_pq_configs) or \
                        HWConfig.is_wildcard_quantization(target_pq_configs):
                    return TransitionStatus.SHOULD_MERGE
                return TransitionStatus.SHOULD_NOT_TRANSITION

            # Check if the target node is affected by any of the quantizers
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                potential_quantizers = from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if potential_quantizers:
                    # Affecting quantizers should have the same configs by construction, so we only
                    # check the first
                    curr_pq_configs = prop_quantizer.potential_quant_configs
                    target_pq_configs = potential_quantizers[0].potential_quant_configs
                    if curr_pq_configs == target_pq_configs or \
                            HWConfig.is_wildcard_quantization(curr_pq_configs) or \
                            HWConfig.is_wildcard_quantization(target_pq_configs):
                        return TransitionStatus.SHOULD_MERGE

                    # Did not merge - the edge will remain untraversed, but the quantizers at the next node will
                    # still be affecting it
                    for pq in potential_quantizers:
                        pq.affected_edges.add((from_node_key, to_node_key))
                        edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(pq)

                    return TransitionStatus.SHOULD_NOT_TRANSITION
        return TransitionStatus.SHOULD_TRANSITION

    def get_merged_qconfigs(self, primary_potential_qconfigs_list: List[QuantizerConfig],
                            secondary_potential_qconfigs_dict: Dict[str, List[QuantizerConfig]]) -> Tuple[
                                List[QuantizerConfig], Dict[str, QuantizerConfig]]:
        """Returns potential qconfigs lists for 'primary' and 'secondary' quantizers
        that are compatible with each other. Compatibility is decided in terms of
        primary quantizer having configs which all have higher precision than all the
        secondary potential quantizer configs."""
        final_primary_merged_qconfigs_list = deepcopy(primary_potential_qconfigs_list)
        curr_secondary_merged_qconfigs_dict = deepcopy(secondary_potential_qconfigs_dict)
        # TODO: implement variant solutions, i.e. for each set of resultant merged
        # primary potential qconfig lists we have, in general, different merged secondary potential
        # config lists. Currently greedy approach is used.
        for m_qconfig in primary_potential_qconfigs_list:
            should_persist_secondary_merged_qconfigs_dict = True
            candidate_secondary_merged_qconfigs_dict = deepcopy(curr_secondary_merged_qconfigs_dict)
            for node_key, s_qconfig_list in curr_secondary_merged_qconfigs_dict.items():
                for s_qconfig in s_qconfig_list:
                    if m_qconfig < s_qconfig and s_qconfig in candidate_secondary_merged_qconfigs_dict[node_key]:
                        candidate_secondary_merged_qconfigs_dict[node_key].remove(s_qconfig)
            for _, s_qconfig_list in candidate_secondary_merged_qconfigs_dict.items():
                if not s_qconfig_list:
                    # No options left for secondary configs on one of the branches to accomodate the primary
                    # config - this primary config cannot be used to be merged into.
                    final_primary_merged_qconfigs_list.remove(m_qconfig)
                    should_persist_secondary_merged_qconfigs_dict = False
                    break
            if should_persist_secondary_merged_qconfigs_dict:
                curr_secondary_merged_qconfigs_dict = candidate_secondary_merged_qconfigs_dict
        if not final_primary_merged_qconfigs_list:
            return [], {}
        return final_primary_merged_qconfigs_list, curr_secondary_merged_qconfigs_dict

    def get_finished_propagating_quantizers(self):
        return self._finished_propagating_quantizers

    def get_active_propagating_quantizers_queue(self):
        return self._active_propagating_quantizers_queue

    def _filter_integer_input_quantizers(self, quant_prop_graph: QuantizerPropagationStateGraph):
        input_node_vs_qid_dict = quant_prop_graph.get_input_quantizer_ids()
        integer_input_quantizer_ids = set()

        for input_node_context, input_quantizer_ids in input_node_vs_qid_dict.items():
            assert input_node_context.operator_name == MODEL_INPUT_OP_NAME
            input_id = input_node_context.call_order
            if self._input_infos[input_id].is_integer_input():
                integer_input_quantizer_ids.update(set(input_quantizer_ids))

        filtered_finished_pqs = list(filter(lambda pq: pq.id not in integer_input_quantizer_ids,
                                            self._finished_propagating_quantizers))
        self._finished_propagating_quantizers = filtered_finished_pqs

    @staticmethod
    def _get_quantizers_between_quantizable_layers_per_node_key(
            quant_prop_graph: QuantizerPropagationStateGraph,
            finished_propagating_quantizers: List[PropagatingQuantizer]):
        visited = {node_key: False for node_key in quant_prop_graph.nodes()}
        quantizers_between_quantizable_layers_per_node_key = {}  # type: Dict[str, QuantizersBetweenQuantizableLayers]

        def traverse_function_up(node_key: str,
                                 output: QuantizersBetweenQuantizableLayers) -> Tuple[bool, Any]:
            if visited[node_key]:
                return True, output
            visited[node_key] = True

            is_finished = False
            node = quant_prop_graph.nodes[node_key]
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]

            if node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[
                    QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    output.add_activation_quantizer_ctx(insertion_point_data.ia_op_exec_context)
                    quantizers_between_quantizable_layers_per_node_key[node_key] = output
                    is_finished = True
                else:
                    for sub_node_key in quant_prop_graph.succ[node_key]:
                        output = quant_prop_graph.traverse_graph(sub_node_key, traverse_function_down, output,
                                                                 traverse_forward=True)
            elif node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                if node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] \
                    == QuantizationTrait.INPUTS_QUANTIZABLE:
                    raise RuntimeError('Should not reach quantizable operator on backward traverse from quantizer!')
            else:
                # reached barrier for nodes in ignored_scopes, no need to go further - this nodes shouldn't be quantized
                is_finished = True
            return is_finished, output

        def traverse_function_down(node_key: str, output: QuantizersBetweenQuantizableLayers) -> Tuple[bool, Any]:
            if visited[node_key]:
                return True, output
            visited[node_key] = True

            node = quant_prop_graph.nodes[node_key]
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            is_finished = False
            if node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[
                    QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    output.add_activation_quantizer_ctx(insertion_point_data.ia_op_exec_context)
                    quantizers_between_quantizable_layers_per_node_key[node_key] = output
                else:
                    for sub_node_key in quant_prop_graph.pred[node_key]:
                        output = quant_prop_graph.traverse_graph(sub_node_key, traverse_function_up, output,
                                                                 traverse_forward=False)
            elif node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                if node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] \
                    == QuantizationTrait.INPUTS_QUANTIZABLE:
                    output.add_quantized_module_scope(node[QuantizerPropagationStateGraph.OPERATOR_SCOPE])
                    is_finished = True
                elif node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] \
                    == QuantizationTrait.NON_QUANTIZABLE:
                    raise RuntimeError('Should not reach non-quantizable operator on forward traverse from quantizer!')
            else:
                # reached barrier for nodes in ignored_scopes, no need to go further - this nodes shouldn't be quantized
                is_finished = True
            return is_finished, output

        for finished_prop_quantizer in finished_propagating_quantizers:
            node_key = finished_prop_quantizer.current_location_node_key
            quantizers_between_quantizable_layers = QuantizersBetweenQuantizableLayers()
            quant_prop_graph.traverse_graph(node_key, traverse_function_down, quantizers_between_quantizable_layers,
                                            traverse_forward=True)

        return quantizers_between_quantizable_layers_per_node_key
