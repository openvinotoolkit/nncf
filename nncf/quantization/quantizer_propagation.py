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
from collections import deque, OrderedDict, Counter
from enum import Enum
from typing import Dict, Tuple, Set, Any, Callable

import networkx as nx
import warnings
from copy import deepcopy

from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import OperationExecutionContext, NNCFGraph, InputAgnosticOperationExecutionContext
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.dynamic_graph.operator_metatypes import *
from nncf.dynamic_graph.operator_metatypes import OPERATOR_METATYPES
from nncf.hw_config import HWConfig
from nncf.dynamic_graph.input_wrapping import MODEL_INPUT_OP_NAME
from nncf.nncf_network import InsertionType, InsertionPointGraph, InsertionPointGraphNodeType, \
    InsertionPoint, InsertionInfo
from nncf.quantization.structs import QuantizationConstraints, QuantizerGroup, QuantizableModule, \
    QuantizersBetweenQuantizableLayers
from nncf.quantization.quantizer_setup import QuantizationPointId, MultiConfigQuantizationPoint, \
    SingleConfigQuantizerSetup, MultiConfigQuantizerSetup
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
        LeakyRELUMetatype,
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
        EmbeddingBagMetatype,
        SoftmaxMetatype
    ],
}  # type: Dict[QuantizationTrait, List[OperatorMetatype]]



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
        self.potential_quant_configs = quant_configs  # type: List[QuantizerConfig]
        self.affected_edges = set()
        self.affected_ip_nodes = set()  # type: Set[str]
        self.propagation_path = []
        self.current_location_node_key = init_location_node_key
        self.last_accepting_location_node_key = None
        self.id = id_
        self.unified_scale = unified_scale
        self.affected_operator_nodes = set()
        self.quantized_input_sink_operator_nodes = set()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)



class TransitionStatus(Enum):
    SHOULD_TRANSITION = 0
    SHOULD_MERGE = 1
    SHOULD_NOT_TRANSITION = 2
    SHOULD_WAIT_FOR_MERGE = 3


class PropagationStrategy(Enum):
    CONSERVATIVE = 0  # While propagating up through a downward-branching node,
                      # do not merge at all, or ...
    MODERATE = 1      # ... only merge for exact matches
    AGGRESSIVE = 2    # ... merge common parts, and if a branch quantizer has options for scope narrowing in addition to
                      # the common part, keep the quantizer on branch


class QuantizerPropagationStateGraphNodeType(Enum):
    INSERTION_POINT = 0
    OPERATOR = 1
    AUXILIARY_BARRIER = 2


class UnifiedScalePropagatingQuantizerGroupManager:
    def __init__(self):
        self._next_gid = 0
        self._group_vs_prop_quants_dict = {}  # type: Dict[int, IPropagatingQuantizerGroup]

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


SharedAffectedOpsQuantizerGroup = QuantizersBetweenQuantizableLayers  # TODO: rename all class references


class SharedAffectedOpsPropagatingQuantizerGroup:
    def __init__(self, affecting_prop_quants: Set[PropagatingQuantizer], affected_op_node_keys: Set[str]):
        self.affecting_prop_quants = affecting_prop_quants  # type: Set[PropagatingQuantizer]
        self.affected_op_node_keys = affected_op_node_keys  # type: Set[str]

    def update(self, other: 'SharedAffectedOpsPropagatingQuantizerGroup'):
        self.affected_op_node_keys.update(other.affected_op_node_keys)
        self.affecting_prop_quants.update(other.affecting_prop_quants)


#pylint:disable=too-many-public-methods
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
    OPERATOR_IA_OP_EXEC_CONTEXT_NODE_ATTR = "ia_op_exec_context"
    IS_IN_IGNORED_SCOPES = "is_ignored"
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
                qpg_node[self.IS_IN_IGNORED_SCOPES] = False

                node_ia_op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic  # type: InputAgnosticOperationExecutionContext
                qpg_node[self.OPERATOR_SCOPE] = node_ia_op_exec_context.scope_in_model
                qpg_node[self.OPERATOR_IA_OP_EXEC_CONTEXT_NODE_ATTR] = node_ia_op_exec_context
                node_scope = str(node_ia_op_exec_context)

                if in_scope_list(node_scope, self._ignored_scopes):
                    qpg_node[self.IS_IN_IGNORED_SCOPES] = True

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
    def merge_quantizer_into_path(self, prop_quantizer: PropagatingQuantizer, path: List[Tuple[str, str]]):
        curr_node = self.nodes[prop_quantizer.current_location_node_key]
        curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        surviving_quantizers = []  # type: List[PropagatingQuantizer]
        for from_node_key, to_node_key in path:
            edge = self.edges[from_node_key, to_node_key]
            edge_affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if edge_affecting_quantizers:
                surviving_quantizers = copy(edge_affecting_quantizers)
                break

            prop_quantizer.affected_edges.add((from_node_key, to_node_key))
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
            from_node = self.nodes[from_node_key]
            node_propagating_quantizer = from_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
            if node_propagating_quantizer is not None:
                surviving_quantizers = [node_propagating_quantizer]
                break
            node_affecting_quantizers = from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if node_affecting_quantizers:
                surviving_quantizers = copy(node_affecting_quantizers)
                break

        if surviving_quantizers:
            for pq in surviving_quantizers:
                pq.affected_operator_nodes.update(prop_quantizer.affected_operator_nodes)
                pq.quantized_input_sink_operator_nodes.update(prop_quantizer.quantized_input_sink_operator_nodes)
                pq.affected_ip_nodes.update(prop_quantizer.affected_ip_nodes)
                pq.affected_edges.update(prop_quantizer.affected_edges)
                for from_node_key, to_node_key in prop_quantizer.affected_edges:
                    to_node = self.nodes[to_node_key]
                    to_node_type = to_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                    if to_node_type in [QuantizerPropagationStateGraphNodeType.INSERTION_POINT,
                                        QuantizerPropagationStateGraphNodeType.OPERATOR]:
                        # pylint:disable=line-too-long
                        self.nodes[to_node_key][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(pq)

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

    def merge_quantizers_for_branching_node(self, quantizers_to_merge: List[PropagatingQuantizer],
                                            merged_qconf_list: List[QuantizerConfig],
                                            branch_qconf_lists: List[List[QuantizerConfig]],
                                            branching_node_key: str):
        assert self.nodes[branching_node_key][QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR] == \
               QuantizerPropagationStateGraphNodeType.INSERTION_POINT

        target_ip_node_key = branching_node_key

        for idx, pq in enumerate(quantizers_to_merge):
            branch_qconf_list = branch_qconf_lists[idx]
            if branch_qconf_list is not None:
                pq.potential_quant_configs = branch_qconf_list

        if merged_qconf_list is None:
            return None

        merge_pq = PropagatingQuantizer(self._get_next_prop_quantizer_id(), merged_qconf_list,
                                        target_ip_node_key)
        merge_pq.last_accepting_location_node_key = target_ip_node_key
        merge_pq.affected_ip_nodes.add(target_ip_node_key)
        target_ip_node = self.nodes[target_ip_node_key]
        assert target_ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is None
        target_ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = merge_pq
        target_ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(merge_pq)

        unified_scale_gids_to_merge = set()
        for idx, pq in enumerate(quantizers_to_merge):
            branch_qconf_list = branch_qconf_lists[idx]
            if branch_qconf_list is None and pq.unified_scale:
                gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(pq.id)
                unified_scale_gids_to_merge.add(gid)

        if unified_scale_gids_to_merge:
            merge_pq.unified_scale = True
            merge_gid = self._unified_scale_group_manager.register_group({merge_pq})
            for gid_to_merge in unified_scale_gids_to_merge:
                self._unified_scale_group_manager.merge_groups(merge_gid, gid_to_merge)

        for idx, pq in enumerate(quantizers_to_merge):
            branch_qconf_list = branch_qconf_lists[idx]
            if branch_qconf_list is None:
                paths = list(nx.all_shortest_paths(self, target_ip_node_key, pq.current_location_node_key))
                assert len(paths) == 1, "Ambiguous merge path!"
                # merge_quantizer_into_path expects paths as lists of edges
                path = paths[0]
                edge_path = []
                for i in range(len(path) - 1):
                    from_node_key = path[i]
                    to_node_key = path[i + 1]
                    edge_path.append((from_node_key, to_node_key))
                self.merge_quantizer_into_path(pq, edge_path)
            else:
                pq.potential_quant_configs = branch_qconf_list

        return merge_pq

    def backtrack_propagation_until_accepting_location(self, prop_quantizer: PropagatingQuantizer) -> \
            Optional[PropagatingQuantizer]:
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
                                  unified_scale: bool = False,
                                  unified_scale_group_id_override: Optional[int] = None) -> PropagatingQuantizer:
        ip_node = self.nodes[ip_node_key]
        ip_type = ip_node[QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR].insertion_type
        if ip_type != InsertionType.OPERATOR_PRE_HOOK:
            # The insertion point key should immediately precede a quantizable op,
            # otherwise it is hard to determine affected node here (although possible)
            raise RuntimeError("Can only add propagating quantizers into pre-hook spots!")

        prop_quantizer = PropagatingQuantizer(self._get_next_prop_quantizer_id(), qconf_list, ip_node_key,
                                              unified_scale)

        if unified_scale:
            if unified_scale_group_id_override is None:
                self._unified_scale_group_manager.register_group({prop_quantizer})
            else:
                self._unified_scale_group_manager.add_to_group(unified_scale_group_id_override,
                                                               prop_quantizer)

        ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        affected_op_node_key = next(self.successors(ip_node_key))
        affected_op_node = self.nodes[affected_op_node_key]
        affected_op_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        initial_edge_key = (ip_node_key, affected_op_node_key)
        initial_edge = self.edges[initial_edge_key]
        initial_edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
        prop_quantizer.affected_edges.add(initial_edge_key)
        prop_quantizer.affected_ip_nodes.add(ip_node_key)
        prop_quantizer.affected_operator_nodes.add(affected_op_node_key)
        prop_quantizer.quantized_input_sink_operator_nodes.add(affected_op_node_key)
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
        for node_key in cloned_prop_quant.affected_operator_nodes:
            node = self.nodes[node_key]
            node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(cloned_prop_quant)

        if cloned_prop_quant.unified_scale:
            gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
            self._unified_scale_group_manager.add_to_group(gid, cloned_prop_quant)

        return cloned_prop_quant

    def remove_propagating_quantizer(self, prop_quantizer: PropagatingQuantizer,
                                     keep_propagating_quantizer_at_current_node=False):
        for edge_tuple in prop_quantizer.affected_edges:
            edge = self.edges[edge_tuple]
            affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)
        for node_key in prop_quantizer.affected_ip_nodes:
            node = self.nodes[node_key]
            affecting_quantizers = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)

        for node_key in prop_quantizer.affected_operator_nodes:
            node = self.nodes[node_key]
            affecting_quantizers = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)

        # No need to handle quantized_input_sink nodes, since these are included in affected_operator_nodes.

        if not keep_propagating_quantizer_at_current_node:
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
            elif from_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
                prop_quantizer.affected_operator_nodes.add(from_node_key)

        target_ip_node_key = path[-1][0]
        prop_quantizer.current_location_node_key = target_ip_node_key
        target_node = self.nodes[target_ip_node_key]
        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        return prop_quantizer

    def get_non_quant_agnostic_op_nodes_immediately_dominated_by_node(self, node_key) -> List[str]:
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

    def get_propagating_quantizers_immediately_dominated_by_node(self, node_key: str) -> Set[PropagatingQuantizer]:
        retval = set()  # type: Set[PropagatingQuantizer]
        def traverse_fn(curr_node_key: str, all_pqs: Set[PropagatingQuantizer]) -> \
                Tuple[bool, Set[PropagatingQuantizer]]:
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                pq = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if pq is not None:
                    all_pqs.add(pq)
                    return True, all_pqs
            return False, all_pqs

        self.traverse_graph(node_key, traverse_fn, retval)
        return retval

    def get_visualized_graph(self):
        out_graph = nx.DiGraph()
        unified_scale_group_vs_pq_node_id_dict = {}  # type: Dict[int, List[str]]
        for node_key, node in self.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[
                    QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                ip_input_port = insertion_point_data.input_port_id
                label = "IP: {}{}".format(insertion_point_data.insertion_type,
                                          (' ' + str(ip_input_port)) if ip_input_port is not None else '')
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
                    quant_node_label += 'Q-input sink ops: {}'.format(
                        "\n".join(prop_quantizer.quantized_input_sink_operator_nodes))
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
        visited_node_keys = set()  # type: Set[str]
        return self._traverse_graph_recursive_helper(curr_node_key,
                                                     visited_node_keys,
                                                     traverse_function, output, traverse_forward)

    def _traverse_graph_recursive_helper(self, curr_node_key: str, visited_node_keys: Set[str],
                                         traverse_function: Callable[[str, Any], Tuple[bool, Any]],
                                         output: Any, traverse_forward: bool):
        is_finished, output = traverse_function(curr_node_key, output)
        visited_node_keys.add(curr_node_key)
        node_keys_holder = self.succ if traverse_forward else self.pred
        if not is_finished:
            for node_key in node_keys_holder[curr_node_key]:
                if node_key not in visited_node_keys:
                    self._traverse_graph_recursive_helper(node_key, visited_node_keys,
                                                          traverse_function, output, traverse_forward)
        return output

    def _get_next_prop_quantizer_id(self):
        self._created_prop_quantizer_counter += 1
        return self._created_prop_quantizer_counter

    def _is_position_accepting(self, ip_node_key: str):
        return True

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

    def merge_redundant_subsequent_quantizers_across_graph(self):
        def is_downstream_quantizer_redundant(downstream_quantizer: PropagatingQuantizer,
                                              upstream_quantizer: PropagatingQuantizer):
            ds_configs = downstream_quantizer.potential_quant_configs
            us_configs = upstream_quantizer.potential_quant_configs
            assert len(ds_configs) == 1
            assert len(us_configs) == 1
            ds_config = ds_configs[0]
            us_config = us_configs[0]
            is_redundant = True
            is_redundant = is_redundant and (ds_config.bits == us_config.bits)

            # Avoid asymmetric quantization if a symmetrically quantized tensor arrived
            is_redundant = is_redundant and ((ds_config.mode == us_config.mode) or (
                ds_config.mode == QuantizationMode.ASYMMETRIC and us_config.mode == QuantizationMode.SYMMETRIC))

            # Avoid per-channel quantization if a per-tensor-quantized tensor arrived
            is_redundant = is_redundant and ((ds_config.per_channel == us_config.per_channel) or (
                ds_config.per_channel is True and us_config.per_channel is False))
            return is_redundant

        def merge_traverse_fn(curr_node_key: str,
                              affecting_pq_and_prev_node_key: Tuple[Optional[PropagatingQuantizer],
                                                                    str]) -> Tuple[Optional[PropagatingQuantizer], str]:
            affecting_pq, prev_node_key = affecting_pq_and_prev_node_key
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                curr_pq = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if curr_pq is not None:
                    if affecting_pq is None:
                        return False, (curr_pq, curr_node_key)

                    if is_downstream_quantizer_redundant(curr_pq, affecting_pq):
                        self.merge_quantizer_into_path(curr_pq, [(prev_node_key, curr_node_key)])
                        return False, (affecting_pq, curr_node_key)

                    return False, (curr_pq, curr_node_key)
            elif curr_node_type == QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER:
                return False, (None, curr_node_key)
            elif curr_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = curr_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait is not QuantizationTrait.QUANTIZATION_AGNOSTIC:
                    return False, (None, curr_node_key)
            return False, (affecting_pq, curr_node_key)

        graph_roots = []
        for node_key in self.nodes:
            if not list(self.predecessors(node_key)):
                graph_roots.append(node_key)

        for graph_root_key in graph_roots:
            self.traverse_graph(graph_root_key, merge_traverse_fn, (None, graph_root_key))

    def collect_all_propagating_quantizers(self) -> Set[PropagatingQuantizer]:
        retval = set()  # type: Set[PropagatingQuantizer]

        def traverse_fn(curr_node_key: str, all_pqs: Set[PropagatingQuantizer]) -> Tuple[
                bool, Set[PropagatingQuantizer]]:
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                pq = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if pq is not None:
                    all_pqs.add(pq)
                    return False, all_pqs
            return False, all_pqs

        graph_roots = []
        for node_key in self.nodes:
            if not list(self.predecessors(node_key)):
                graph_roots.append(node_key)

        for graph_root_key in graph_roots:
            self.traverse_graph(graph_root_key, traverse_fn, retval)

        return retval

    def get_insertion_point_for_propagating_quantizer(self, prop_quant: PropagatingQuantizer) -> InsertionPoint:
        final_node_key = prop_quant.current_location_node_key
        final_node = self.nodes[final_node_key]
        insertion_point = final_node[
            QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
        return insertion_point

    def _get_all_quantizers_grouped_by_affecting_op_set(self) -> List[SharedAffectedOpsPropagatingQuantizerGroup]:
        all_pqs = self.collect_all_propagating_quantizers()

        class Grouper:
            """Propagating quantizers will be grouped so that each quantizer is in the same group as the
            node that it is affecting. Furthermore, each quantizer that does not affect any node
            (e.g. if it only affects other quantizers as a topmost quantizer in a requantization
            scenario) will be placed in a separate group."""

            def __init__(self):
                self._group_vs_node_keys_and_pqs = {}  # type: Dict[int, SharedAffectedOpsPropagatingQuantizerGroup]
                self._next_gid = 0

            def _get_next_gid(self):
                curr_gid = self._next_gid
                self._next_gid += 1
                return curr_gid

            def _merge_groups(self, gid_to: int, gid_from: int):
                self._group_vs_node_keys_and_pqs[gid_to].update(self._group_vs_node_keys_and_pqs[gid_from])
                self._group_vs_node_keys_and_pqs.pop(gid_from)

            def add_pq(self, pq: PropagatingQuantizer):
                new_gid = self._get_next_gid()
                self._group_vs_node_keys_and_pqs[new_gid] = \
                    SharedAffectedOpsPropagatingQuantizerGroup({pq}, set(pq.quantized_input_sink_operator_nodes))
                new_group_data = self._group_vs_node_keys_and_pqs[new_gid]
                gids_to_merge = set()  # type: Set[int]
                for gid, group_data in self._group_vs_node_keys_and_pqs.items():
                    if gid == new_gid:
                        continue
                    for node_key in new_group_data.affected_op_node_keys:
                        if node_key in group_data.affected_op_node_keys:
                            gids_to_merge.add(gid)

                for gid_to_merge in gids_to_merge:
                    self._merge_groups(new_gid, gid_to_merge)

            def get_groups(self) -> Dict[int, SharedAffectedOpsPropagatingQuantizerGroup]:
                return self._group_vs_node_keys_and_pqs

        grouper = Grouper()
        for pq in all_pqs:
            grouper.add_pq(pq)

        groups = grouper.get_groups()
        return list(groups.values())

    def get_num_input_activations(self, operator_node_key: str) -> int:
        assert self.nodes[operator_node_key][QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR] == \
               QuantizerPropagationStateGraphNodeType.OPERATOR
        return len(list(self.predecessors(operator_node_key)))

    def create_quantizer_setup(self, quantizable_module_scope_vs_qconfigs: Dict[Scope, List[QuantizerConfig]]) \
            -> 'MultiConfigQuantizerSetup':
        same_op_groups = self._get_all_quantizers_grouped_by_affecting_op_set()
        setup = MultiConfigQuantizerSetup()

        qm_scope_vs_same_op_group_idx_in_list = {}  # type: Dict[Scope, int]
        for group in same_op_groups:
            grouped_ids = set()
            for pq in group.affecting_prop_quants:
                quant_point = MultiConfigQuantizationPoint(self.get_insertion_point_for_propagating_quantizer(pq),
                                                           pq.potential_quant_configs)
                setup.quantization_points[pq.id] = quant_point
                grouped_ids.add(pq.id)

            setup.shared_input_operation_set_groups.append(grouped_ids)
            group_idx_in_list = len(setup.shared_input_operation_set_groups) - 1
            for module_scope in quantizable_module_scope_vs_qconfigs.keys():
                # Hacky, but this is all we can do since the graph we are dealing with has merged nodes,
                # and the only information about which nodes were there before the merge is in the node key,
                # which is a concatenation of the original node keys.
                # TODO: preserve pre-merge node keys information explicitly
                for affected_node_key in group.affected_op_node_keys:
                    if str(module_scope) in affected_node_key:
                        qm_scope_vs_same_op_group_idx_in_list[module_scope] = group_idx_in_list

        if setup.quantization_points.keys():
            max_aq_id = max(setup.quantization_points.keys()) + 1
        else:
            max_aq_id = 0
        next_wq_id = max_aq_id + 1
        for module_scope, qconfig_list in quantizable_module_scope_vs_qconfigs.items():
            insertion_point = InsertionPoint(InsertionType.NNCF_MODULE_PRE_OP,
                                             module_scope=module_scope)
            quant_point = MultiConfigQuantizationPoint(insertion_point, qconfig_list)
            setup.quantization_points[next_wq_id] = quant_point
            if module_scope not in qm_scope_vs_same_op_group_idx_in_list:
                # Happens for LSTM cells. The "hidden" Linear layer, as represented in NNCFGraph, has no
                # input edges, since its input is not a regular network input, but a recurrent input
                # from the previous execution step. TODO: extend recurrent operations handling so that NNCF graph
                # has information on which operation accepts recurrent inputs.
                nncf_logger.warning("Could not find an associated input activation quantizer for a module with "
                                    "quantizable weights: {}\n".format(module_scope))
            else:
                associated_same_op_group_list_idx = qm_scope_vs_same_op_group_idx_in_list[module_scope]
                setup.shared_input_operation_set_groups[associated_same_op_group_list_idx].add(next_wq_id)
            next_wq_id += 1

        pq_sets_grouped_by_unified_scale = list(
            self._unified_scale_group_manager.get_group_vs_prop_quants_dict().values())
        for pq_set in pq_sets_grouped_by_unified_scale:
            setup.unified_scale_groups.append({pq.id for pq in pq_set})

        return setup

    def run_consistency_check(self) -> bool:
        all_pqs = self.collect_all_propagating_quantizers()

        def traverse_fn(curr_node_key: str, unused) -> Tuple[bool, Any]:
            nncf_logger.debug("Processing node: {}".format(curr_node_key))
            node = self.nodes[curr_node_key]
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                pq = node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]  # type: PropagatingQuantizer
                affecting_pqs = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if pq is not None:
                    assert pq in affecting_pqs
                    assert pq.current_location_node_key == curr_node_key
                for affecting_pq in affecting_pqs:
                    assert affecting_pq in all_pqs
                    assert curr_node_key in affecting_pq.affected_ip_nodes
            elif node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                affecting_pqs = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                for affecting_pq in affecting_pqs:
                    assert affecting_pq in all_pqs
                    assert curr_node_key in affecting_pq.affected_operator_nodes

            for out_edge_key in self.out_edges(curr_node_key):
                nncf_logger.debug("Processing edge: {}".format(out_edge_key))
                out_edge = self.edges[out_edge_key]
                affecting_pqs = out_edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                for pq in affecting_pqs:
                    assert pq in all_pqs
                    assert out_edge_key in pq.affected_edges
            return False, None

        graph_roots = []
        for node_key in self.nodes:
            if not list(self.predecessors(node_key)):
                graph_roots.append(node_key)

        for graph_root_key in graph_roots:
            self.traverse_graph(graph_root_key, traverse_fn, None)

        for pq in all_pqs:
            location_node = self.nodes[pq.current_location_node_key]
            assert pq == location_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
            for edge_key in pq.affected_edges:
                edge = self.edges[edge_key]
                assert pq in edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            for edge_key in pq.propagation_path:
                edge = self.edges[edge_key]
                assert pq in edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            for affected_ip_node_key in pq.affected_ip_nodes:
                ip_node = self.nodes[affected_ip_node_key]
                assert ip_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR] == \
                       QuantizerPropagationStateGraphNodeType.INSERTION_POINT
                assert pq in ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            for affected_op_node_key in pq.affected_operator_nodes:
                op_node = self.nodes[affected_op_node_key]
                assert op_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR] == \
                       QuantizerPropagationStateGraphNodeType.OPERATOR
                assert pq in op_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]


class QuantizersWaitingForMergeManager:
    """Tracks the quantizers that await a merge while trying to transition through a downward-branching node
    and corresponding node keys."""

    def __init__(self):
        self._branching_node_keys_vs_quantizers_waiting_for_merge = {}  # type: Dict[str, Set[PropagatingQuantizer]]
        self._quantizers_vs_branching_node_keys = {}  # type: Dict[PropagatingQuantizer, str]

    def add_propagating_quantizer_to_wait_on_node_key(self, pq: PropagatingQuantizer, branching_node_key: str):
        if branching_node_key not in self._branching_node_keys_vs_quantizers_waiting_for_merge:
            self._branching_node_keys_vs_quantizers_waiting_for_merge[branching_node_key] = set()
        self._branching_node_keys_vs_quantizers_waiting_for_merge[branching_node_key].add(pq)
        self._quantizers_vs_branching_node_keys[pq] = branching_node_key

    def get_blocking_node(self, pq: PropagatingQuantizer) -> str:
        return self._quantizers_vs_branching_node_keys[pq]

    def get_waiting_quantizers_for_branching_node_key(self, node_key: str) -> Set[PropagatingQuantizer]:
        return self._branching_node_keys_vs_quantizers_waiting_for_merge[node_key]

    def __contains__(self, item: PropagatingQuantizer):
        return item in self._quantizers_vs_branching_node_keys.keys()

    def resolve_merged_node(self, branching_node_key: str):
        for pq in self._branching_node_keys_vs_quantizers_waiting_for_merge[branching_node_key]:
            self._quantizers_vs_branching_node_keys.pop(pq)
        self._branching_node_keys_vs_quantizers_waiting_for_merge.pop(branching_node_key)


class FinalizedQuantizationProposal:
    def __init__(self, single_config_quantizer_setup: SingleConfigQuantizerSetup,
                 quant_prop_graph: QuantizerPropagationStateGraph):
        self.single_config_quantizer_setup = single_config_quantizer_setup
        self._quant_prop_graph = quant_prop_graph

    @property
    def quant_prop_graph(self):
        return self._quant_prop_graph


class QuantizationProposal:
    def __init__(self, quantizer_setup: 'MultiConfigQuantizerSetup',
                 quant_prop_graph: QuantizerPropagationStateGraph,
                 quantization_point_id_vs_prop_quantizer: Dict['QuantizationPointId',
                                                               PropagatingQuantizer]):
        self.quantizer_setup = quantizer_setup
        self._quant_prop_graph = quant_prop_graph
        self._quantization_point_id_vs_prop_quantizer = quantization_point_id_vs_prop_quantizer
        self._prop_quantizer_vs_quantization_point_id = {}  # type: Dict[PropagatingQuantizer, QuantizationPointId]
        for qp_id, pq in self._quantization_point_id_vs_prop_quantizer.items():
            self._prop_quantizer_vs_quantization_point_id[pq] = qp_id

    def constrain_quantizer_config_list_for_insertion(self, quantization_point_id: 'QuantizationPointId',
                                                      constrained_config_list: List[QuantizerConfig]):
        prior_list = self.quantizer_setup.quantization_points[quantization_point_id].possible_qconfigs
        if not all([qc in prior_list for qc in constrained_config_list]):
            raise RuntimeError("Constrained config list is incompatible with the result of the quantizer propagation!")
        # TODO: only allow to constrain "input-group"-wise?
        self.quantizer_setup.quantization_points[quantization_point_id].possible_qconfigs = constrained_config_list

        if quantization_point_id in self._quantization_point_id_vs_prop_quantizer:
            pq = self._quantization_point_id_vs_prop_quantizer[quantization_point_id]
            pq.potential_quant_configs = constrained_config_list

    def finalize(self, final_quantizer_setup: SingleConfigQuantizerSetup, strict=True):
        for pq, qp_id in self._prop_quantizer_vs_quantization_point_id.items():
            if qp_id not in final_quantizer_setup.quantization_points:
                self._quant_prop_graph.remove_propagating_quantizer(pq)
            else:
                final_qconfig = final_quantizer_setup.quantization_points[qp_id].qconfig
                if strict:
                    def is_final_qconfig_compatible_to_initial(initial_qconfig: QuantizerConfig):
                        return final_qconfig.per_channel == initial_qconfig.per_channel and \
                               final_qconfig.mode == initial_qconfig.mode and \
                               final_qconfig.bits == initial_qconfig.bits and \
                               (final_qconfig.signedness_to_force == initial_qconfig.signedness_to_force or
                                initial_qconfig.signedness_to_force is None or
                                final_qconfig.signedness_to_force is None)

                    compatible_initial_qconfs = list(
                        filter(is_final_qconfig_compatible_to_initial,
                               self.quantizer_setup.quantization_points[qp_id].possible_qconfigs))
                    if not compatible_initial_qconfs:
                        raise RuntimeError("The final quantizer setup has configurations that were not present in the "
                                           "initial proposal!")
                    if final_qconfig.signedness_to_force is None:
                        initial_qconfs_signedness_values = {qc.signedness_to_force for qc in compatible_initial_qconfs}
                        if None not in initial_qconfs_signedness_values and len(initial_qconfs_signedness_values) == 1:
                            # The initial configs were either all forced-signed or all forced-unsigned - should set
                            # final qconfig's forced field appropriately
                            final_qconfig.signedness_to_force = initial_qconfs_signedness_values.pop()

                pq.potential_quant_configs = [final_qconfig]
        return FinalizedQuantizationProposal(final_quantizer_setup,
                                             self._quant_prop_graph)


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
                 input_infos: List[ModelInputInfo] = None,
                 quantizable_modules: List[QuantizableModule] = None,
                 scope_overrides: Dict = None,
                 global_constraints: Dict[QuantizerGroup, QuantizationConstraints] = None,
                 run_consistency_checks: bool = False):
        self._quantizers_between_quantizable_layers_per_key = {}  # type: Dict[str, QuantizersBetweenQuantizableLayers]
        self.default_qlobal_qconfig_list = default_qconfig_list
        self._hw_config = hw_config  # type: HWConfig
        self._debug_interface = debug_interface
        self._propagation_strategy = propagation_strategy  # TODO: determine from config
        self._operator_quantization_trait_map = self.get_operator_quantization_traits_map()
        self._operator_allowed_qconfigs_map = self._get_operator_qconfigs_map()
        self._input_infos = input_infos
        self._quantizable_modules = quantizable_modules  # type: List['QuantizableModule']
        if scope_overrides is None:
            self._scope_overrides = {}
        else:
            self._scope_overrides = scope_overrides  # type: Dict
        self._global_constraints = global_constraints  # type: Dict['QuantizerGroup', 'QuantizationConstraints']
        self._run_consistency_checks = run_consistency_checks

        if self._hw_config is not None:
            self._unified_scales_operation_set = self._hw_config.get_operations_with_unified_scales()
        else:
            self._unified_scales_operation_set = {}

        # Will handle the "wildcard" quantization situation for the time being
        if default_qconfig_list is not None:
            for op_meta, qconf_list in self._operator_allowed_qconfigs_map.items():
                trait = self._operator_quantization_trait_map[op_meta]
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    if HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconf_list):
                        self._operator_allowed_qconfigs_map[op_meta] = default_qconfig_list
        self._active_propagating_quantizers_queue = deque()
        self._finished_propagating_quantizers = []  # type: List[PropagatingQuantizer]
        self._ignored_scopes = ignored_scopes
        self._quantizers_waiting_for_branch_merge = QuantizersWaitingForMergeManager()

        self._potential_quantizers = {}
        self._num_potential_quantized_activations = 0

    def run_on_ip_graph(self, ip_graph: InsertionPointGraph) -> QuantizationProposal:
        """ The main function to be used on an InsertionPointGraph to produce
            the list of insertion commands and configs corresponding to the final quantized
            graph state."""
        self._num_potential_quantized_activations = 0
        quant_prop_graph = QuantizerPropagationStateGraph(ip_graph, self._ignored_scopes)
        quant_prop_graph = self.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
        quant_prop_graph = self.setup_initial_quantizers(quant_prop_graph)

        if self._run_consistency_checks:
            quant_prop_graph.run_consistency_check()

        iteration_counter = 0
        while self._active_propagating_quantizers_queue:
            if self._debug_interface is not None:
                self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, str(iteration_counter))
            if self._run_consistency_checks:
                quant_prop_graph.run_consistency_check()
            prop_quantizer = self._active_propagating_quantizers_queue.pop()
            quant_prop_graph = self.propagation_step(prop_quantizer, quant_prop_graph)
            iteration_counter += 1

        if self._input_infos is not None:
            quant_prop_graph = self._filter_integer_input_quantizers(quant_prop_graph)

        if self._debug_interface is not None:
            self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, "proposed")

        if self._run_consistency_checks:
            quant_prop_graph.run_consistency_check()

        if self._quantizable_modules is not None:
            quantizable_module_scopes_vs_qconfig_list_dict = {qm.module_scope: qm.qconfig_list for qm in
                                                              self._quantizable_modules}
        else:
            quantizable_module_scopes_vs_qconfig_list_dict = {}

        quantizer_setup = quant_prop_graph.create_quantizer_setup(quantizable_module_scopes_vs_qconfig_list_dict)
        insertions_vs_associated_prop_quants = self._map_quantization_points_to_prop_quantizers(
            self._finished_propagating_quantizers, quant_prop_graph, quantizer_setup)

        return QuantizationProposal(quantizer_setup=quantizer_setup,
                                    quant_prop_graph=quant_prop_graph,
                                    quantization_point_id_vs_prop_quantizer=insertions_vs_associated_prop_quants)

    def _map_quantization_points_to_prop_quantizers(self,
                                                    prop_quant_list: List[PropagatingQuantizer],
                                                    quant_prop_graph: QuantizerPropagationStateGraph,
                                                    quantizer_setup: MultiConfigQuantizerSetup) -> \
            Dict[QuantizationPointId, PropagatingQuantizer]:
        qps_vs_associated_prop_quants_dict = {}  # type: Dict[QuantizationPointId, PropagatingQuantizer]

        for finished_prop_quantizer in prop_quant_list:
            insertion_point = quant_prop_graph.get_insertion_point_for_propagating_quantizer(finished_prop_quantizer)
            for qp_id, qp in quantizer_setup.quantization_points.items():
                if qp.insertion_point == insertion_point:
                    qps_vs_associated_prop_quants_dict[qp_id] = finished_prop_quantizer

        return qps_vs_associated_prop_quants_dict

    def get_final_quantizer_setup(self, finalized_quantization_proposal: FinalizedQuantizationProposal) -> \
            SingleConfigQuantizerSetup:
        """Merges consequent quantizers which ended up having the same quantization configuration."""
        quant_prop_graph = finalized_quantization_proposal.quant_prop_graph
        quant_prop_graph.merge_redundant_subsequent_quantizers_across_graph()

        if self._debug_interface is not None:
            self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, "final")

        if self._run_consistency_checks:
            quant_prop_graph.run_consistency_check()

        quantizable_module_scopes_vs_qconfig_list_dict = {qm.module_scope: qm.qconfig_list for qm in
                                                          self._quantizable_modules}
        final_module_scopes_vs_qconfig_dict = {}
        for qp in finalized_quantization_proposal.single_config_quantizer_setup.quantization_points.values():
            if qp.is_weight_quantization_point():
                final_module_scopes_vs_qconfig_dict[qp.insertion_point.module_scope] = [qp.qconfig]  # sic!

        if Counter(final_module_scopes_vs_qconfig_dict.keys()) != \
                Counter(quantizable_module_scopes_vs_qconfig_list_dict.keys()):
            raise RuntimeError("Final weight quantizer setup is inconsistent with initial solver assumptions!")

        multi_setup_with_one_config_per_point = quant_prop_graph.create_quantizer_setup(
            final_module_scopes_vs_qconfig_dict)
        final_setup = multi_setup_with_one_config_per_point.select_first_qconfig_for_each_point()
        return final_setup

    def get_num_potential_quantized_activations(self) -> int:
        return self._num_potential_quantized_activations

    def _get_insertion_info_for_propagating_quantizer(self, prop_quant: PropagatingQuantizer,
                                                      quant_prop_graph: QuantizerPropagationStateGraph) -> \
            InsertionInfo:

        insertion_point = quant_prop_graph.get_insertion_point_for_propagating_quantizer(prop_quant)
        in_port_id = None
        if insertion_point.insertion_type == InsertionType.OPERATOR_PRE_HOOK:
            in_port_id = insertion_point.input_port_id

        op_exec_context = OperationExecutionContext(
            operator_name=insertion_point.ia_op_exec_context.operator_name,
            scope_in_model=insertion_point.ia_op_exec_context.scope_in_model,
            call_order=insertion_point.ia_op_exec_context.call_order,
            tensor_metas=[None]  # TODO: fix this, rethink InsertionInfo here and elsewhere
        )
        insertion_info = InsertionInfo(op_exec_context, in_port_id=in_port_id)
        return insertion_info

    def _handle_quantizer_merge(self, waiting_pqs: Set[PropagatingQuantizer],
                                quant_prop_graph: QuantizerPropagationStateGraph,
                                branching_node_key: str):
        waiting_pqs_list = list(waiting_pqs)
        # All quantizers that are dominated by the current branching node are waiting
        # for the merge - should merge them now
        nncf_logger.debug("Merging PQs: {}".format(",".join([str(pq.id) for pq in waiting_pqs_list])))
        qconfs_list = [pq.potential_quant_configs for pq in waiting_pqs_list]
        merged_qconf_list, branch_qconf_lists = \
            self.get_merged_qconfigs_for_downward_branching_case(qconfs_list)

        merge_pq = quant_prop_graph.merge_quantizers_for_branching_node(waiting_pqs_list,
                                                                        merged_qconf_list,
                                                                        branch_qconf_lists,
                                                                        branching_node_key)

        merged_pqs = []
        unmerged_pqs = []

        for idx, qconf_list in enumerate(branch_qconf_lists):
            if qconf_list is None:
                merged_pqs.append(waiting_pqs_list[idx])
            else:
                unmerged_pqs.append(waiting_pqs_list[idx])

        queue_to_cull = list(reversed(self._active_propagating_quantizers_queue))
        self._active_propagating_quantizers_queue.clear()
        for pq_from_queue in queue_to_cull:
            if pq_from_queue in merged_pqs:
                continue
            if pq_from_queue in unmerged_pqs:
                finished_pq = quant_prop_graph.backtrack_propagation_until_accepting_location(pq_from_queue)
                if finished_pq is not None:
                    self._finished_propagating_quantizers.append(pq_from_queue)
            else:
                self._active_propagating_quantizers_queue.appendleft(pq_from_queue)

        if merge_pq is not None:
            self._active_propagating_quantizers_queue.appendleft(merge_pq)
        self._quantizers_waiting_for_branch_merge.resolve_merged_node(branching_node_key)

    def propagation_step(self, curr_prop_quantizer: PropagatingQuantizer,
                         quant_prop_graph: QuantizerPropagationStateGraph) -> QuantizerPropagationStateGraph:
        """Returns an updated curr_prop_quantizer state if the quantizer is not
           yet in its final (accepting) position, and None if the quantizer is in its
           final location.  The location before and after the step should correspond to
           some insertion point."""
        # TODO: full-fledged discrete finite automata approach? Switch to traversing a graph
        # consisting of insertion points only, with reversed edges holding associated operator data?

        #pylint:disable=too-many-branches
        curr_node_key = curr_prop_quantizer.current_location_node_key
        curr_node = quant_prop_graph.nodes[curr_prop_quantizer.current_location_node_key]
        curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
        assert curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT

        if curr_prop_quantizer in self._quantizers_waiting_for_branch_merge:
            branching_node_key = self._quantizers_waiting_for_branch_merge.get_blocking_node(curr_prop_quantizer)
            dom_pqs = quant_prop_graph.get_propagating_quantizers_immediately_dominated_by_node(branching_node_key)
            active_dom_pqs = set(
                filter(lambda x: x in self._active_propagating_quantizers_queue or x is curr_prop_quantizer, dom_pqs))
            waiting_pqs = self._quantizers_waiting_for_branch_merge.get_waiting_quantizers_for_branching_node_key(
                branching_node_key)
            if waiting_pqs == active_dom_pqs:
                self._active_propagating_quantizers_queue.append(curr_prop_quantizer)
                self._handle_quantizer_merge(waiting_pqs, quant_prop_graph, branching_node_key)
            else:
                # Not all of the dominated quantizers have reached the blocking node yet
                self._active_propagating_quantizers_queue.appendleft(curr_prop_quantizer)
            return quant_prop_graph

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
        did_clone = False

        for _ in range(1, len(paths)):
            additional_prop_quantizer = quant_prop_graph.clone_propagating_quantizer(curr_prop_quantizer)
            prop_quantizers_to_process.append(additional_prop_quantizer)
            did_clone = True

        cloned_prop_quantizers = prop_quantizers_to_process if did_clone else None

        pqs_and_paths = zip(paths, prop_quantizers_to_process)
        for path, prop_quantizer in pqs_and_paths:
            status = self.check_transition_via_path(prop_quantizer, path, quant_prop_graph,
                                                    cloned_prop_quantizers)
            if status == TransitionStatus.SHOULD_NOT_TRANSITION:
                if did_clone and prop_quantizer is not curr_prop_quantizer:
                    quant_prop_graph.remove_propagating_quantizer(prop_quantizer,
                                                                  keep_propagating_quantizer_at_current_node=True)
                else:
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
            elif status == TransitionStatus.SHOULD_WAIT_FOR_MERGE:
                branching_node_key = None
                for from_node_key, _ in path:
                    if len(list(quant_prop_graph.successors(from_node_key))) > 1:
                        branching_node_key = path[0][0]
                        break
                assert branching_node_key is not None
                #pylint:disable=line-too-long
                self._quantizers_waiting_for_branch_merge.add_propagating_quantizer_to_wait_on_node_key(prop_quantizer,
                                                                                                        branching_node_key)
                surviving_prop_quantizers.append(prop_quantizer)

        for prop_quantizer in surviving_prop_quantizers:
            self._active_propagating_quantizers_queue.appendleft(prop_quantizer)
        return quant_prop_graph

    def get_allowed_quantizer_configs_for_operator(self, quant_det_id: OperatorMetatype) -> List[QuantizerConfig]:
        return self._operator_allowed_qconfigs_map[quant_det_id]

    def set_allowed_quantization_types_for_operator_nodes(self, quant_prop_graph: QuantizerPropagationStateGraph):
        for node_key, node in quant_prop_graph.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                if node[QuantizerPropagationStateGraph.IS_IN_IGNORED_SCOPES]:
                    trait = QuantizationTrait.NON_QUANTIZABLE
                    node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] = trait
                    continue

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
        next_id_str = ""
        if self._active_propagating_quantizers_queue:
            next_id_str = str(self._active_propagating_quantizers_queue[-1].id)
        out_graph.graph['graph'] = {
            "label": "Propagating quantizers: {}\n" \
                     "Next quantizer to be propagated: {}\n" \
                     "Finished quantizers: {}".format(active_ids_str,
                                                      next_id_str,
                                                      finished_ids_str),
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
                num_input_activations = quant_prop_graph.get_num_input_activations(node_key)
                self._num_potential_quantized_activations += num_input_activations
                if node_key in quant_prop_graph.ignored_node_keys:
                    nncf_logger.info("Ignored adding Activation input quantizer for: {}".format(node_key))
                    continue
                self._setup_initial_quantizers_for_operator_node(node_key, quant_prop_graph)

        return quant_prop_graph

    def _filter_qconfigs_according_to_scope(self, qconf_list: List[QuantizerConfig],
                                            operator_scope_str: str) -> List[QuantizerConfig]:
        if self._global_constraints is not None:
            local_constraints = self._global_constraints[QuantizerGroup.ACTIVATIONS]
        else:
            local_constraints = QuantizationConstraints()
        for overridden_scope, scoped_override_dict in self._scope_overrides.items():
            if in_scope_list(operator_scope_str, overridden_scope):
                scope_constraints = QuantizationConstraints.from_config_dict(scoped_override_dict)
                local_constraints = local_constraints.get_updated_constraints(scope_constraints)

        if self._hw_config is not None:
            try:
                constrained_config_list = local_constraints.constrain_qconfig_list(qconf_list)
            except RuntimeError as e:
                err_msg = "Quantization parameter constraints specified in NNCF config are incompatible with HW "
                err_msg += "capabilities as specified in HW config type '{}'. ".format(self._hw_config.target_device)
                err_msg += "First conflicting quantizer location: "
                err_msg += operator_scope_str
                raise RuntimeError(err_msg) from e
        else:
            constrained_config_list = [local_constraints.apply_constraints_to(qconfig) for qconfig in qconf_list]

        return constrained_config_list

    def _setup_initial_quantizers_for_operator_node(self, operator_node_key: str,
                                                    quant_prop_graph: QuantizerPropagationStateGraph):
        node = quant_prop_graph.nodes[operator_node_key]
        preds = list(quant_prop_graph.predecessors(operator_node_key))

        if not preds:
            return  # TODO: remove this once module insertion points are included in the IP graph

        # No need to place quantizers for FP32-forced ops, naturally
        if node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] in \
                [QuantizationTrait.NON_QUANTIZABLE,
                 QuantizationTrait.QUANTIZATION_AGNOSTIC]:
            return

        quant_det_id = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
        qconf_list = self.get_allowed_quantizer_configs_for_operator(quant_det_id)
        assert qconf_list is not None

        ia_op_exec_context = node[QuantizerPropagationStateGraph.OPERATOR_IA_OP_EXEC_CONTEXT_NODE_ATTR]
        if not HWConfig.is_wildcard_quantization(qconf_list):
            qconf_list = self._filter_qconfigs_according_to_scope(qconf_list, str(ia_op_exec_context))
        else:
            from nncf.quantization.algo import QuantizerSetupGeneratorBase
            qconf_list = [deepcopy(QuantizerSetupGeneratorBase.DEFAULT_QUANTIZER_CONFIG)]

        is_unified_scale = quant_det_id in self._unified_scales_operation_set
        if is_unified_scale:
            # Filtering out the per-channel cases in the unified scale scenario.
            # In order to support unified per-channel scales, we will need to handle a situation
            # when a per-channel shared (unified) quantizer on one branch passes a shape-changing
            # operation such as `view` or `transpose`, and the linked unified quantizer does not do
            # so due to one or the other reason; the per-channel scale shapes to be applied therefore
            # will be different.
            # TODO: What possibly needs to be done in this direction:
            # 1. keep ForwardTraceOnly ops in graph after all, to be able to track shape changes
            # 2. transpose input tensors to the quantization modules on the fly to accommodate scale,
            #    or vice versa, transpose scale to accommodate shape; need to handle exporting as well
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

        pred_ip_key_vs_qconf_dict = OrderedDict()
        # Should be immediately preceded by insertion points (pre-hook)
        for pred_ip_key in preds:
            pred_node = quant_prop_graph.nodes[pred_ip_key]
            pred_node_type = pred_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            assert pred_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT, \
                "Invalid insertion point graph supplied for quantizer propagation!"

            pred_ip_key_vs_qconf_dict[pred_ip_key] = qconf_list

        # Cloning a single propagating quantizer onto all node inputs - revise if separate
        # quantizer configuration for different inputs is required
        pred_ip_key_vs_qconf_list = list(iter(pred_ip_key_vs_qconf_dict.items()))
        main_pq_ip_key, main_pq_qconf_list = pred_ip_key_vs_qconf_list[0]
        main_prop_quantizer = quant_prop_graph.add_propagating_quantizer(main_pq_qconf_list, main_pq_ip_key,
                                                                         is_unified_scale)
        main_prop_quantizer.last_accepting_location_node_key = main_pq_ip_key
        self._active_propagating_quantizers_queue.appendleft(main_prop_quantizer)

        main_pq_gid = None

        if is_unified_scale:
            main_pq_gid = quant_prop_graph.get_unified_scale_group_id_by_propagating_quantizer_id(
                main_prop_quantizer.id)

        for additional_pq_ip_key, _ in pred_ip_key_vs_qconf_list[1:]:
            additional_pq = quant_prop_graph.add_propagating_quantizer(main_pq_qconf_list,
                                                                       additional_pq_ip_key,
                                                                       unified_scale=is_unified_scale,
                                                                       unified_scale_group_id_override=main_pq_gid)
            additional_pq.last_accepting_location_node_key = additional_pq_ip_key
            self._active_propagating_quantizers_queue.appendleft(additional_pq)

    # pylint:disable=too-many-return-statements
    def check_branching_transition(self, quant_prop_graph: QuantizerPropagationStateGraph,
                                   prop_quant_to_transition: PropagatingQuantizer,
                                   branching_node_key: str) -> Optional[TransitionStatus]:
        """If a propagating quantizer advances through a node that branches
           downwards, the branches neighbouring to the one that the propagating quantizer
           had just propagated from will have the precision of the quantizer imposed upon
           them.  This is not always desirable - we might want to keep some branches in
           higher precision than the others. For this reason, this function checks whether
           the quantizer may safely advance through a branching node based on the possible
           configs of the quantizers it might affect by doing so."""
        dom_op_node_keys = quant_prop_graph.get_non_quant_agnostic_op_nodes_immediately_dominated_by_node(
            branching_node_key)
        dom_op_quantizers = set()
        for op_node_key in dom_op_node_keys:
            op_node = quant_prop_graph.nodes[op_node_key]
            affecting_prop_quantizers = op_node[
                QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if not affecting_prop_quantizers:
                # The branch op is forced to be FP32 - should not proceed through the branch node.
                return TransitionStatus.SHOULD_NOT_TRANSITION
            for aff_pq in affecting_prop_quantizers:
                dom_op_quantizers.add(aff_pq)

        dom_op_quantizers.discard(prop_quant_to_transition)
        if dom_op_quantizers:
            return TransitionStatus.SHOULD_WAIT_FOR_MERGE

        return TransitionStatus.SHOULD_TRANSITION

    def _check_affecting_quantizers_in_common_path(self,
                                                   affecting_quantizers: List[PropagatingQuantizer],
                                                   cloned_prop_quantizers: List[PropagatingQuantizer]):
        # Handling the case where multiple freshly cloned quantizers have to follow paths that are different,
        # but have a common edge or node
        safe_affecting_quantizers = [pq for pq in affecting_quantizers if pq in cloned_prop_quantizers]
        assert safe_affecting_quantizers == affecting_quantizers

    def _check_for_affecting_quantizer_conflicts(self,
                                                 curr_prop_quantizer: PropagatingQuantizer,
                                                 affecting_quantizers: List[PropagatingQuantizer],
                                                 cloned_prop_quantizers: Optional[List[PropagatingQuantizer]]
                                                 ) -> Optional[TransitionStatus]:
        if cloned_prop_quantizers is not None:
            self._check_affecting_quantizers_in_common_path(affecting_quantizers, cloned_prop_quantizers)
            return None

        # Affecting quantizers should have the same configs by construction, so we only
        # check the first
        curr_pq_configs = curr_prop_quantizer.potential_quant_configs
        target_pq_configs = affecting_quantizers[0].potential_quant_configs
        if curr_pq_configs == target_pq_configs or \
                HWConfig.is_wildcard_quantization(curr_pq_configs) or \
                HWConfig.is_wildcard_quantization(target_pq_configs):
            return TransitionStatus.SHOULD_MERGE
        return TransitionStatus.SHOULD_NOT_TRANSITION

    def check_transition_via_path(self, prop_quantizer: PropagatingQuantizer, path: List,
                                  quant_prop_graph: QuantizerPropagationStateGraph,
                                  cloned_prop_quantizers: Optional[
                                      List[PropagatingQuantizer]] = None) -> TransitionStatus:
        """Determines which action should be taken regarding the
           prop_quantizer's propagation via path, which may be one of many possible
           propagation paths."""
        for from_node_key, to_node_key in path:
            from_node = quant_prop_graph.nodes[from_node_key]

            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = from_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait in [QuantizationTrait.NON_QUANTIZABLE,
                             QuantizationTrait.INPUTS_QUANTIZABLE]:
                    return TransitionStatus.SHOULD_NOT_TRANSITION

            # Check if current edge to traverse is affected by any of the quantizers
            edge = quant_prop_graph.edges[from_node_key, to_node_key]
            potential_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if potential_quantizers:
                sts = self._check_for_affecting_quantizer_conflicts(prop_quantizer,
                                                                    potential_quantizers,
                                                                    cloned_prop_quantizers)

                if sts is not None:
                    return sts

            # Check if the target node is affected by any of the quantizers
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                potential_quantizers = from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if potential_quantizers:
                    sts = self._check_for_affecting_quantizer_conflicts(prop_quantizer,
                                                                        potential_quantizers,
                                                                        cloned_prop_quantizers)
                    if sts == TransitionStatus.SHOULD_NOT_TRANSITION:

                        # Did not merge - the edge will remain untraversed, but the quantizers at the next node will
                        # still be affecting it
                        for pq in potential_quantizers:
                            pq.affected_edges.add((from_node_key, to_node_key))
                            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(pq)

                        return sts

                    if sts is not None:
                        return sts

            if len(list(quant_prop_graph.successors(from_node_key))) > 1:
                # If a quantizer simply passes up through a downward-branching node, it may spoil the
                # precision for operations on neighbouring branches. Consider a 4-bit quantizer rising
                # through a branch node and an 8-bit quantizer arriving at the same node later. Therefore,
                # prior to allowing the quantizer to pass through a branching node we need to ensure that
                # the precision of the quantizer is a superset of precisions of the first non-quantization agnostic
                # operations on each branch.
                status = self.check_branching_transition(quant_prop_graph, prop_quantizer,
                                                         from_node_key)
                if status is TransitionStatus.SHOULD_NOT_TRANSITION or status is TransitionStatus.SHOULD_WAIT_FOR_MERGE:
                    return status

        return TransitionStatus.SHOULD_TRANSITION

    def get_merged_qconfigs_for_downward_branching_case(self,
                                                        potential_qconfigs_for_each_branch: List[
                                                            List[Optional[QuantizerConfig]]]) -> \
            Tuple[Optional[List[QuantizerConfig]], List[List[Optional[QuantizerConfig]]]]:
        """Returns a tuple, of which the first element is the qconfig list for the quantizer to be placed
        above the branching node (i.e. that will affect all of the downward branches), and a list
        of elements which are either None (which means that the corresponding branch quantizer has been successfully
        merged, or qconfigs list to be set for the corresponding branch quantizer if it cannot be merged (e.g. if
        requantization to a lower bitwidth has to be done for this branch)"""
        #pylint:disable=too-many-branches
        if self._propagation_strategy == PropagationStrategy.CONSERVATIVE:
            # Do not merge at all
            return None, potential_qconfigs_for_each_branch
        if self._propagation_strategy == PropagationStrategy.MODERATE:
            # Only merge for exact matches of the qconfig lists
            first_pq_list = potential_qconfigs_for_each_branch[0]
            first_pq_list_counter = Counter(first_pq_list)

            for other_pq_list in potential_qconfigs_for_each_branch[1:]:
                if first_pq_list_counter != Counter(other_pq_list):
                    return None, potential_qconfigs_for_each_branch
            return first_pq_list, [None for _ in potential_qconfigs_for_each_branch]

        # Aggressive
        qconfigs_union = set()
        for branch_qconfig_list in potential_qconfigs_for_each_branch:
            qconfigs_union.update(set(branch_qconfig_list))
        merged_qconfig_list = []

        def compatible(qconf, other_qconf_list):
            if qconf in other_qconf_list:
                return True
            for other_qconf in other_qconf_list:
                if qconf < other_qconf:
                    return False
            return True

        nncf_logger.debug("Union of configs: {}".format(";".join([str(qc) for qc in qconfigs_union])))
        for qconf in qconfigs_union:
            if all([compatible(qconf, qconf_list) for qconf_list in potential_qconfigs_for_each_branch]):
                merged_qconfig_list.append(qconf)

        merged_qconfig_list_counter = Counter(merged_qconfig_list)
        resulting_branch_qconfig_lists = [None for _ in potential_qconfigs_for_each_branch]
        for idx, branch_qconfig_list in enumerate(potential_qconfigs_for_each_branch):
            if Counter(branch_qconfig_list) == merged_qconfig_list_counter:
                continue  # This branch will have the branch quantizer removed
            resulting_branch_qconfig_lists[idx] = branch_qconfig_list

        nncf_logger.debug("Merged list before sorting: {}".format(";".join([str(qc) for qc in merged_qconfig_list])))

        if not merged_qconfig_list:
            merged_qconfig_list = None

        if merged_qconfig_list is not None:
            # Sort the merged list according to an ad-hoc-calculated priority
            # Basically, the original branches vote on a priority of a config in the merged
            # qconfig list based on the position of said qconfig in their own qconfig list.
            # TODO: This still does not properly disambiguate configs in all situations. Downstream code
            # takes 0-th config in the list as the final config file. Without an external, unambiguous
            # priority mechanism or manual config selection there is no way to do a consistent, branch order-independent
            # merge.
            qconfig_and_priority_list = []  # type: List[Tuple[QuantizerConfig, int]]
            for merged_qconfig in merged_qconfig_list:
                priority = 0
                max_original_list_len = max([len(x) for x in potential_qconfigs_for_each_branch])
                for original_branch_qconfig_list in potential_qconfigs_for_each_branch:
                    try:
                        idx = original_branch_qconfig_list.index(merged_qconfig)
                    except ValueError:
                        # Move the configs that inevitably lead to requantization closer to the end of the list
                        idx = max_original_list_len + 1
                    priority += idx
                qconfig_and_priority_list.append((merged_qconfig, priority))

            qconfig_and_priority_list_sorted_by_priority = sorted(qconfig_and_priority_list, key=lambda x: x[1])
            nncf_logger.debug(
                "Priority-sorted merge qconfigs: {}".format(";".join(
                    [str(qc_tup[1]) + ':' + str(qc_tup[0]) for qc_tup in
                     qconfig_and_priority_list_sorted_by_priority])))

            merged_qconfig_list = self.__disambiguate_config_list(qconfig_and_priority_list_sorted_by_priority)
            nncf_logger.debug(
                "Disambiguated merge qconfig list: {}".format(";".join([str(qc) for qc in merged_qconfig_list])))

        return merged_qconfig_list, resulting_branch_qconfig_lists

    def __disambiguate_config_list(self, qconfig_list_with_priority: List[Tuple[QuantizerConfig, int]]) -> \
            List[QuantizerConfig]:
        """The input list should be sorted in descending order of priority. In case some qconfigs in the list have the
        same priority, this function will resolve the ambiguity in ordering these qconfigs in the final returned
        list."""
        class QConfigComparator:
            def __init__(self, qconfig: QuantizerConfig):
                self.qconfig = qconfig

            def __lt__(self, other: 'QConfigComparator'):
                # Prefer higher bitwidths, per-tensor, symmetrical
                if self.qconfig.bits > other.qconfig.bits:
                    return True
                if self.qconfig.bits < other.qconfig.bits:
                    return False
                if self.qconfig.per_channel is False and other.qconfig.per_channel is True:
                    return True
                if self.qconfig.per_channel is True and other.qconfig.per_channel is False:
                    return False
                if self.qconfig.mode is QuantizationMode.SYMMETRIC and other.qconfig.mode is \
                        QuantizationMode.ASYMMETRIC:
                    return True
                if self.qconfig.mode is QuantizationMode.ASYMMETRIC and other.qconfig.mode is \
                        QuantizationMode.SYMMETRIC:
                    return False
                return False

        slices_to_sort = []

        if len(qconfig_list_with_priority) > 1:
            curr_priority_start_idx = 0
            curr_priority = qconfig_list_with_priority[0][1]
            for idx, val in enumerate(qconfig_list_with_priority):
                if val[1] != curr_priority:
                    if (idx - curr_priority_start_idx) > 1:
                        slices_to_sort.append(slice(curr_priority_start_idx, idx))
                    curr_priority_start_idx = idx
                    curr_priority = val[1]

            last_idx = len(qconfig_list_with_priority) - 1
            if last_idx - curr_priority_start_idx > 0:
                slices_to_sort.append(slice(curr_priority_start_idx, last_idx + 1))

        list_to_sort = [QConfigComparator(x[0]) for x in qconfig_list_with_priority]
        for slice_obj in slices_to_sort:
            list_to_sort[slice_obj] = sorted(list_to_sort[slice_obj])

        retval = [x.qconfig for x in list_to_sort]
        return retval

    def get_finished_propagating_quantizers(self):
        return self._finished_propagating_quantizers

    def get_active_propagating_quantizers_queue(self):
        return self._active_propagating_quantizers_queue

    def get_total_quantizer_count(self):
        return len(self.get_finished_propagating_quantizers()) + len(self.get_active_propagating_quantizers_queue())

    def _filter_integer_input_quantizers(self,
                                         quant_prop_graph: QuantizerPropagationStateGraph) -> \
            QuantizerPropagationStateGraph:
        input_node_vs_qid_dict = quant_prop_graph.get_input_quantizer_ids()
        integer_input_quantizer_ids = set()

        for input_node_context, input_quantizer_ids in input_node_vs_qid_dict.items():
            assert input_node_context.operator_name == MODEL_INPUT_OP_NAME
            input_id = input_node_context.call_order
            if self._input_infos[input_id].is_integer_input():
                integer_input_quantizer_ids.update(set(input_quantizer_ids))

        filtered_finished_pqs = list(filter(lambda pq: pq.id not in integer_input_quantizer_ids,
                                            self._finished_propagating_quantizers))
        integer_input_pqs = list(filter(lambda pq: pq.id in integer_input_quantizer_ids,
                                        self._finished_propagating_quantizers))
        self._finished_propagating_quantizers = filtered_finished_pqs
        for integer_input_pq in integer_input_pqs:
            quant_prop_graph.remove_propagating_quantizer(integer_input_pq)

        return quant_prop_graph

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
                    # Reached another FQ for activation by going down & up from a FQ for activation.
                    # here->|FQ_A|  Conv  start_FQ_A    Conv
                    #         \   /           |       /
                    #        POST_HOOK       POST_HOOK
                    #              \        /
                    #                Concat
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
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    # Reached another FQ for activation by going down from a FQ for activation.
                    # Should be processed within another group
                    # start->FQ_A  Conv
                    #         \   /
                    #        POST_HOOK
                    #          /    \
                    #   PRE_HOOK    PRE_HOOK
                    #     |           \
                    #   div          MaxPool   here->|FQ_A|
                    #                   \     /
                    #                 POST_HOOK
                    visited[node_key] = False
                    is_finished = True
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
            # process PropagatingQuantizer separately by adding FQ for activation to the group and by starting downward
            # traverse for all child nodes. No need to check upward branches as it's final location of the quantizer
            if not visited[node_key]:
                quantizers_between_quantizable_layers = QuantizersBetweenQuantizableLayers()
                node = quant_prop_graph.nodes[node_key]
                insertion_point_data = node[
                    QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    visited[node_key] = True
                    quantizers_between_quantizable_layers.add_activation_quantizer_insertion_point(
                        insertion_point_data)
                for next_node_key in quant_prop_graph.succ[node_key]:
                    quant_prop_graph.traverse_graph(next_node_key, traverse_function_down,
                                                    quantizers_between_quantizable_layers, traverse_forward=True)
                quantizers_between_quantizable_layers_per_node_key[node_key] = quantizers_between_quantizable_layers
        return quantizers_between_quantizable_layers_per_node_key
