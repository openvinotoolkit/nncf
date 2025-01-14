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

from collections import deque
from copy import copy
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Type, Union

import networkx as nx

import nncf
from nncf import nncf_logger
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NoopMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import InsertionPointGraphNodeType
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from nncf.common.quantization.quantizer_propagation.grouping import UnifiedScalePropagatingQuantizerGroupManager
from nncf.common.quantization.quantizer_propagation.structs import IgnoreReason
from nncf.common.quantization.quantizer_propagation.structs import PropagatingQuantizer
from nncf.common.quantization.quantizer_propagation.structs import PropagationPath
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationStateGraphNodeType
from nncf.common.quantization.quantizer_propagation.structs import SharedAffectedOpsPropagatingQuantizerGroup
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import QuantizationInsertionPointBase
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import UnifiedScaleType
from nncf.common.scopes import should_consider_scope


class QuantizerPropagationStateGraph(nx.DiGraph):
    """
    This class is based upon InsertionPointGraph and represents
    a"chessboard" for PropagatingQuantizer items.  It tracks the current state of
    quantizer propagation by associating the operator and insertion point nodes and
    edges to propagating quantizers, if any. It can move a propagating quantizer
    via own edges and mark its progress through the graph, which is required for
    resolving situations when multiple quantizers attempt to proceed via one and
    the same graph node/edge. This class is mainly operated upon by the
    QuantizerPropagationSolver objects.
    """

    PROPAGATING_QUANTIZER_NODE_ATTR = "propagating_quantizer"
    AFFECTING_PROPAGATING_QUANTIZERS_ATTR = "affecting_propagating_quantizers"
    QUANTIZATION_TRAIT_NODE_ATTR = "quantization_trait"
    ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR = "allowed_input_quantization_types"
    OPERATOR_METATYPE_NODE_ATTR = "op_meta"
    QUANT_INSERTION_POINT_DATA_NODE_ATTR = "quant_insertion_point"
    NODE_TYPE_NODE_ATTR = "node_type"
    IS_IN_IGNORED_SCOPES = "is_ignored"
    IS_MERGED_NODE_ATTR = "is_merged"
    MERGED_NNCF_NODE_LIST_NODE_ATTR = "merged_node_list"
    IS_INTEGER_PATH_EDGE_ATTR = "is_integer"
    BARRIER_NODE_KEY_POSTFIX = "BARRIER"

    def __init__(
        self,
        ip_graph: InsertionPointGraph,
        ignored_scopes: Dict[str, IgnoreReason] = None,
        target_scopes: List[str] = None,
    ):
        super().__init__()
        ip_graph = deepcopy(ip_graph)
        self._created_prop_quantizer_counter = 0

        self._ignored_scopes = list(ignored_scopes.keys()) if ignored_scopes is not None else None
        self._target_scopes = deepcopy(target_scopes)
        self.ignored_node_keys: Dict[str, IgnoreReason] = {}

        self._unified_scale_group_manager = UnifiedScalePropagatingQuantizerGroupManager()
        self._input_node_keys_vs_nncf_nodes: Dict[str, NNCFNode] = {}
        self._output_node_keys_vs_nncf_nodes: Dict[str, NNCFNode] = {}
        self._pqs_after_weight_dependent_output_quantized_nodes: Dict[PropagatingQuantizer, str] = {}
        self.op_node_keys_to_underlying_nodes_mapping: Dict[str, List[NNCFNode]] = {}

        iteration_scope_node_keys = []
        for node_key, node in ip_graph.nodes.items():
            qpg_node = {
                self.NODE_TYPE_NODE_ATTR: self.ipg_node_type_to_qpsg_node_type(
                    node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
                )
            }
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] in [
                InsertionPointGraphNodeType.PRE_HOOK,
                InsertionPointGraphNodeType.POST_HOOK,
            ]:
                qpg_node[self.PROPAGATING_QUANTIZER_NODE_ATTR] = None
                qpg_node[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []

                ip = node[InsertionPointGraph.INSERTION_POINT_NODE_ATTR]
                qip = self._insertion_point_to_quant_insertion_point(ip)
                qpg_node[self.QUANT_INSERTION_POINT_DATA_NODE_ATTR] = qip

            elif node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                qpg_node[self.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR] = set()
                qpg_node[self.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.NON_QUANTIZABLE
                qpg_node[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
                qpg_node[self.IS_IN_IGNORED_SCOPES] = False

                nncf_node_ref: NNCFNode = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]

                qpg_node[self.IS_MERGED_NODE_ATTR] = node[InsertionPointGraph.IS_MERGED_NODE_ATTR]
                if node[InsertionPointGraph.IS_MERGED_NODE_ATTR]:
                    underlying_nncf_nodes = node[InsertionPointGraph.MERGED_NNCF_NODE_LIST_NODE_ATTR]
                else:
                    underlying_nncf_nodes = [node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]]
                assert underlying_nncf_nodes
                self.op_node_keys_to_underlying_nodes_mapping[node_key] = underlying_nncf_nodes

                ignored = False
                # For the fused-pattern nodes, will only ignore the entire pattern if the primary node in the
                # merged pattern is in the ignored scopes. The primary node is the first one in the
                # underlying_nncf_nodes list.
                primary_node = underlying_nncf_nodes[0]
                if not should_consider_scope(primary_node.node_name, self._ignored_scopes, self._target_scopes):
                    ignored = True

                if ignored:
                    qpg_node[self.IS_IN_IGNORED_SCOPES] = True
                    self.ignored_node_keys[node_key] = ignored_scopes.get(
                        primary_node.node_name, IgnoreReason.USER_REQUESTED
                    )
                    # TODO (vshampor): do we need here NoopMetatype
                    qpg_node[self.OPERATOR_METATYPE_NODE_ATTR] = NoopMetatype
                else:
                    qpg_node[self.OPERATOR_METATYPE_NODE_ATTR] = nncf_node_ref.metatype

                if nncf_node_ref.metatype in INPUT_NOOP_METATYPES:
                    self._input_node_keys_vs_nncf_nodes[node_key] = nncf_node_ref
                if nncf_node_ref.metatype in OUTPUT_NOOP_METATYPES:
                    self._output_node_keys_vs_nncf_nodes[node_key] = nncf_node_ref

                if nncf_node_ref.is_in_iteration_scope():
                    iteration_scope_node_keys.append(node_key)

            self.add_node(node_key, **qpg_node)

        for from_node, to_node, edge_data in ip_graph.edges(data=True):
            edge_data[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
            is_integer = edge_data.pop(InsertionPointGraph.IS_INTEGER_PATH_EDGE_ATTR)
            edge_data[self.IS_INTEGER_PATH_EDGE_ATTR] = is_integer
            self.add_edge(from_node, to_node, **edge_data)

        for barred_node_key in list(self.ignored_node_keys.keys()) + iteration_scope_node_keys:
            self._add_barrier_after_node(barred_node_key)
        self._branch_nodes_directly_dominating_outputs = None

    def get_input_node_keys(self) -> List[str]:
        """
        Returns graph input node keys.

        :return: List of the input node keys.
        """
        return self._input_node_keys_vs_nncf_nodes.keys()

    def get_node_keys_by_metatype(self, metatype: Type[OperatorMetatype]) -> List[str]:
        """
        Returns a list of node keys, whose metatype is corresponding to the 'metatype'.

        :param metatype: The metatype to look for.
        :return: List of node keys.
        """
        output = []
        for node, node_metatype in self.nodes(self.OPERATOR_METATYPE_NODE_ATTR):
            if node_metatype == metatype:
                output.append(node)
        return output

    @staticmethod
    def _insertion_point_to_quant_insertion_point(
        ip: Union[PreHookInsertionPoint, PostHookInsertionPoint]
    ) -> QuantizationInsertionPointBase:
        if isinstance(ip, PreHookInsertionPoint):
            return ActivationQuantizationInsertionPoint(ip.target_node_name, input_port_id=ip.input_port_id)
        assert isinstance(ip, PostHookInsertionPoint)
        return ActivationQuantizationInsertionPoint(ip.target_node_name, input_port_id=None)

    def _add_barrier_after_node(self, node_key: str):
        qpg_node_barrier = {
            self.NODE_TYPE_NODE_ATTR: QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER,
            "label": QuantizerPropagationStateGraph.BARRIER_NODE_KEY_POSTFIX,
        }
        barrier_node_key = self.get_barrier_node_key(node_key)
        self.add_node(barrier_node_key, **qpg_node_barrier)

        next_node_keys = list(self.succ[node_key].keys())
        for next_node_key in next_node_keys:
            edge_attrs = self.edges[node_key, next_node_key]
            self.add_edge(node_key, barrier_node_key, **edge_attrs)
            self.add_edge(barrier_node_key, next_node_key, **edge_attrs)
            self.remove_edge(node_key, next_node_key)

    @staticmethod
    def ipg_node_type_to_qpsg_node_type(
        ipg_node_type: InsertionPointGraphNodeType,
    ) -> QuantizerPropagationStateGraphNodeType:
        if ipg_node_type == InsertionPointGraphNodeType.PRE_HOOK:
            return QuantizerPropagationStateGraphNodeType.PRE_HOOK
        if ipg_node_type == InsertionPointGraphNodeType.POST_HOOK:
            return QuantizerPropagationStateGraphNodeType.POST_HOOK
        if ipg_node_type == InsertionPointGraphNodeType.OPERATOR:
            return QuantizerPropagationStateGraphNodeType.OPERATOR
        raise nncf.ValidationError("Invalid insertion point graph node type.")

    @staticmethod
    def get_barrier_node_key(node_key: str) -> str:
        return f"{QuantizerPropagationStateGraph.BARRIER_NODE_KEY_POSTFIX} {node_key}"

    def mark_act_quantizer_as_dependent_on_weights(self, pq: PropagatingQuantizer, operator_node_key: str):
        """
        Marks a given propagating quantizer corresponding to input activation quantization
        of some downstream op as dependent on weights of an operation that gives its weights directly
        as outputs (such as Embedding). The quantizer marked in this manner will be later considered
        for removal if the weights of the weight-as-outputs operation are quantized in a compatible
        way (i.e. with the same quantizer configuration) as is required by the propagating activation
        quantizer.

        :param: pq - the propagating quantizer corresponding to input quantization of some op
        :param: operator_node_key - a key of the node in QuantizerPropagationStateGraph that corresponds to
            a weights-as-outputs node.
        """
        op_node = self.nodes[operator_node_key]
        assert (
            op_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            is QuantizerPropagationStateGraphNodeType.OPERATOR
        )
        assert (
            op_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
            is QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS
        )
        if (
            pq in self._pqs_after_weight_dependent_output_quantized_nodes
            and self._pqs_after_weight_dependent_output_quantized_nodes[pq] != operator_node_key
        ):
            raise nncf.InternalError(
                f"Propagating quantizer {pq.id} is already marked as depending on node "
                f"{operator_node_key} weight quantization!"
            )
        self._pqs_after_weight_dependent_output_quantized_nodes[pq] = operator_node_key

    @staticmethod
    def is_insertion_point(qpsg_node_type: QuantizerPropagationStateGraphNodeType) -> bool:
        return qpsg_node_type in [
            QuantizerPropagationStateGraphNodeType.PRE_HOOK,
            QuantizerPropagationStateGraphNodeType.POST_HOOK,
        ]

    def merge_quantizer_into_path(self, prop_quantizer: PropagatingQuantizer, path: PropagationPath):
        curr_node = self.nodes[prop_quantizer.current_location_node_key]
        curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        surviving_quantizers: List[PropagatingQuantizer] = []
        for from_node_key, to_node_key in path:
            edge = self.edges[from_node_key, to_node_key]
            edge_affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if edge_affecting_quantizers:
                surviving_quantizers = copy(edge_affecting_quantizers)
                break

            prop_quantizer.affected_edges.add((from_node_key, to_node_key))
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
            from_node = self.nodes[from_node_key]
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(from_node_type):
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
                if prop_quantizer in pq.downstream_propagating_quantizers:
                    pq.downstream_propagating_quantizers.remove(prop_quantizer)
                for from_node_key, to_node_key in prop_quantizer.affected_edges:
                    to_node = self.nodes[to_node_key]
                    to_node_type = to_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                    if to_node_type in [
                        QuantizerPropagationStateGraphNodeType.PRE_HOOK,
                        QuantizerPropagationStateGraphNodeType.POST_HOOK,
                        QuantizerPropagationStateGraphNodeType.OPERATOR,
                    ]:
                        self.nodes[to_node_key][
                            QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR
                        ].append(pq)

            if prop_quantizer.unified_scale_type is not None:
                gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
                for other_pq in surviving_quantizers:
                    if other_pq.unified_scale_type is not None:
                        other_gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(
                            other_pq.id
                        )
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
            raise nncf.InternalError(
                "Surviving_quantizers not found !"
                " Nodes quantized with quantizer #{} will be lost".format(prop_quantizer.id)
            )

    @staticmethod
    def _get_major_unified_scale_type(type_list: List[Optional[UnifiedScaleType]]) -> Optional[UnifiedScaleType]:
        """
        Treats input list entries as unified scale types of merged quantizers, and outputs
        the unified scale type of the resulting merge-quantizer so that it is still compatible with the
        downstream ops.
        """
        major_unified_scale_type = None
        if UnifiedScaleType.UNIFY_ALWAYS in type_list:
            major_unified_scale_type = UnifiedScaleType.UNIFY_ALWAYS
        if UnifiedScaleType.UNIFY_ONLY_PER_TENSOR in type_list:
            major_unified_scale_type = UnifiedScaleType.UNIFY_ONLY_PER_TENSOR
        return major_unified_scale_type

    def merge_quantizers_for_branching_node(
        self,
        quantizers_to_merge: List[PropagatingQuantizer],
        merged_qconf_list: List[QuantizerConfig],
        branch_qconf_lists: List[Optional[List[QuantizerConfig]]],
        branching_node_key: str,
    ) -> List[PropagatingQuantizer]:
        # A branching node may currently be either a post-hook node, or an operator node if the
        # corresponding operator does not support post-hooking (such as torch.chunk)
        branching_node_type = self.nodes[branching_node_key][QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]

        target_ip_node_keys = []
        if self.is_insertion_point(branching_node_type):
            target_ip_node_keys.append(branching_node_key)
        elif branching_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
            paths = self.get_paths_to_immediately_dominating_insertion_points(branching_node_key)
            for path in paths:
                assert len(path) == 1
                edge_from_pre_hook_ip_to_op = path[0]
                pre_hook_ip = edge_from_pre_hook_ip_to_op[0]
                target_ip_node_keys.append(pre_hook_ip)
        else:
            raise nncf.InternalError("Unsupported branching QPSG node type: {}".format(branching_node_type))

        if not target_ip_node_keys:
            return []

        for idx, pq in enumerate(quantizers_to_merge):
            branch_qconf_list = branch_qconf_lists[idx]
            if branch_qconf_list is not None:
                pq.potential_quant_configs = branch_qconf_list

        if merged_qconf_list is None:
            return []

        unified_scale_types_of_merged_branches = [
            pq.unified_scale_type for idx, pq in enumerate(quantizers_to_merge) if branch_qconf_lists[idx] is None
        ]
        merge_pq_unified_scale_type = self._get_major_unified_scale_type(unified_scale_types_of_merged_branches)

        merge_gid = None
        if merge_pq_unified_scale_type is not None:
            merge_gid = self._unified_scale_group_manager.register_group(set())

        merge_pqs = []
        for target_ip_node_key in target_ip_node_keys:
            target_ip_node = self.nodes[target_ip_node_key]
            target_type = target_ip_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if target_type is QuantizerPropagationStateGraphNodeType.PRE_HOOK:
                merge_pq = self.add_propagating_quantizer(
                    merged_qconf_list,
                    target_ip_node_key,
                    unified_scale_type=merge_pq_unified_scale_type,
                    unified_scale_group_id_override=merge_gid,
                )
            elif target_type is QuantizerPropagationStateGraphNodeType.POST_HOOK:
                merge_pq = PropagatingQuantizer(
                    self._get_next_prop_quantizer_id(),
                    merged_qconf_list,
                    target_ip_node_key,
                    unified_scale_type=merge_pq_unified_scale_type,
                )
                merge_pq.last_accepting_location_node_key = target_ip_node_key
                merge_pq.affected_ip_nodes.add(target_ip_node_key)

                target_ip_node = self.nodes[target_ip_node_key]
                assert target_ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is None
                target_ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = merge_pq
                target_ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(merge_pq)
                if merge_gid is not None:
                    self._unified_scale_group_manager.add_to_group(merge_gid, merge_pq)
            else:
                raise nncf.InternalError("Unsupported target type for merge PQ insertion: {}".format(target_type))

            merge_pqs.append(merge_pq)

        unified_scale_gids_to_merge = set()
        for idx, pq in enumerate(quantizers_to_merge):
            branch_qconf_list = branch_qconf_lists[idx]
            if branch_qconf_list is None and pq.unified_scale_type is not None:
                gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(pq.id)
                unified_scale_gids_to_merge.add(gid)

        if unified_scale_gids_to_merge:
            assert merge_gid is not None
            for gid_to_merge in unified_scale_gids_to_merge:
                self._unified_scale_group_manager.merge_groups(merge_gid, gid_to_merge)

        for idx, pq in enumerate(quantizers_to_merge):
            branch_qconf_list = branch_qconf_lists[idx]
            if branch_qconf_list is None:
                paths = list(nx.all_shortest_paths(self, branching_node_key, pq.current_location_node_key))
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
                for merge_pq in merge_pqs:
                    merge_pq.downstream_propagating_quantizers.add(pq)

            # The quantizer sink node set of the merge PQ should be set to the union of all
            # downstream quantizers regardless of whether the downstream PQ has been completely merged
            for merge_pq in merge_pqs:
                merge_pq.quantized_input_sink_operator_nodes.update(pq.quantized_input_sink_operator_nodes)

        return merge_pqs

    def get_predecessor_weight_as_outputs_node_keys(self, curr_node_key: str) -> List[str]:
        """
        For a given node key in this graph, returns node keys of all direct predecessors
        of this node that correspond to weights-as-outputs operations (such as Embedding)

        :param: curr_node_key - a node key in this QuantizerPropagationStateGraph
        :return: A list of weights-as-outputs predecessor node keys for `curr_node_key`
        """
        pred_keys = list(self.predecessors(curr_node_key))
        matches = []
        for pred_key in pred_keys:
            pred_node = self.nodes[pred_key]
            pred_node_type = pred_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if pred_node_type is QuantizerPropagationStateGraphNodeType.OPERATOR:
                pred_node_trait = pred_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if pred_node_trait is QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS:
                    matches.append(pred_key)
        return matches

    def backtrack_propagation_until_accepting_location(
        self, prop_quantizer: PropagatingQuantizer
    ) -> Optional[PropagatingQuantizer]:
        if prop_quantizer.last_accepting_location_node_key is None:
            # The quantizer was stillborn.
            # If there are quantizer-affected inbound edges, should transfer this quantizer's
            # affected edges and nodes to the inbound edge quantizers
            curr_node_key = prop_quantizer.current_location_node_key
            inbound_affecting_quantizers = set()
            for in_edge_key in self.in_edges(curr_node_key):
                in_edge = self.edges[in_edge_key]
                inbound_affecting_quantizers.update(
                    in_edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                )

            for inbound_pq in inbound_affecting_quantizers:
                inbound_pq.affected_edges.update(prop_quantizer.affected_edges)
                inbound_pq.affected_ip_nodes.update(prop_quantizer.affected_ip_nodes)
            for edge in prop_quantizer.affected_edges:
                self.edges[edge][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] += list(
                    inbound_affecting_quantizers
                )
            for ip_node_key in prop_quantizer.affected_ip_nodes:
                self.nodes[ip_node_key][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] += list(
                    inbound_affecting_quantizers
                )

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
            if self.is_insertion_point(from_node_type):
                from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].remove(prop_quantizer)
                prop_quantizer.affected_ip_nodes.remove(from_node_key)

            to_node = self.nodes[to_node_key]
            to_node_type = to_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(to_node_type):
                prop_quantizer.current_location_node_key = to_node_key

        target_ip_node_key = prop_quantizer.current_location_node_key
        target_node = self.nodes[target_ip_node_key]
        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        return prop_quantizer

    def unify_pq_scales(
        self,
        primary_pq: PropagatingQuantizer,
        secondary_pq: PropagatingQuantizer,
        unified_scale_type: Optional[UnifiedScaleType] = None,
    ):
        if unified_scale_type is None:
            primary_pq.unified_scale_type = UnifiedScaleType.UNIFY_ALWAYS
        else:
            primary_pq.unified_scale_type = unified_scale_type
        secondary_pq.unified_scale_type = primary_pq.unified_scale_type
        primary_gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(primary_pq.id)
        if primary_gid is None:
            primary_gid = self._unified_scale_group_manager.register_group({primary_pq})
        self._unified_scale_group_manager.add_to_group(primary_gid, secondary_pq)

    def add_propagating_quantizer(
        self,
        qconf_list: List[QuantizerConfig],
        ip_node_key: str,
        unified_scale_type: Optional[UnifiedScaleType] = None,
        unified_scale_group_id_override: Optional[int] = None,
    ) -> PropagatingQuantizer:
        ip_node = self.nodes[ip_node_key]
        ip_type = ip_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
        if ip_type != QuantizerPropagationStateGraphNodeType.PRE_HOOK:
            # The insertion point key should immediately precede a quantizable op,
            # otherwise it is hard to determine affected node here (although possible)
            raise nncf.InternalError("Can only add propagating quantizers into pre-hook spots!")

        prop_quantizer = PropagatingQuantizer(
            self._get_next_prop_quantizer_id(), qconf_list, ip_node_key, unified_scale_type
        )

        if unified_scale_type is not None:
            if unified_scale_group_id_override is None:
                self._unified_scale_group_manager.register_group({prop_quantizer})
            else:
                self._unified_scale_group_manager.add_to_group(unified_scale_group_id_override, prop_quantizer)

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

    def _verify_nodes_and_edges_for_pq(self, prop_quantizer: PropagatingQuantizer):
        node_keys_to_verify = (
            list(prop_quantizer.affected_operator_nodes)
            + list(prop_quantizer.quantized_input_sink_operator_nodes)
            + [prop_quantizer.current_location_node_key]
            + list(prop_quantizer.affected_ip_nodes)
        )
        if prop_quantizer.last_accepting_location_node_key is not None:
            node_keys_to_verify.append(prop_quantizer.last_accepting_location_node_key)

        for node_key in node_keys_to_verify:
            if node_key not in self.nodes:
                raise nncf.InternalError(
                    "Unknown node referenced by propagating quantizer to be registered: {}".format(node_key)
                )
        edge_keys_to_verify = list(prop_quantizer.affected_edges) + list(prop_quantizer.propagation_path)
        for edge_key in edge_keys_to_verify:
            if edge_key not in self.edges:
                raise nncf.InternalError(
                    "Unknown edge referenced by propagating quantizer to be registered: {}".format(edge_key)
                )

    @staticmethod
    def _verify_qconfig_matching(
        prop_quantizer: PropagatingQuantizer, existing_prop_quantizers: List[PropagatingQuantizer]
    ):
        for existing_pq in existing_prop_quantizers:
            if existing_pq.potential_quant_configs != prop_quantizer.potential_quant_configs:
                raise nncf.InternalError(
                    "Configurations of the quantizer to be registered are conflicting with "
                    "existing quantizer {}".format(existing_pq.id)
                )

    def register_propagating_quantizer(self, prop_quantizer: PropagatingQuantizer):
        """Will only succeed if the new quantizer information is consistent with the rest of the graph state."""
        all_pqs = self.collect_all_propagating_quantizers()
        for existing_pq_id in all_pqs:
            if prop_quantizer.id == existing_pq_id:
                raise nncf.InternalError(
                    "The propagating quantizer to be registered has an ID that is already assigned to "
                    "an existing propagating quantizer!"
                )
        target_node = self.nodes[prop_quantizer.current_location_node_key]
        pq_in_target_node = target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
        if pq_in_target_node is not None:
            raise nncf.InternalError(
                "The propagating quantizer to be registered is occupying the same position "
                "as an existing propagating quantizer {}!".format(pq_in_target_node.id)
            )
        target_node_affecting_quantizers = target_node[
            QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR
        ]
        if target_node_affecting_quantizers:
            raise nncf.InternalError(
                "Cannot register a propagating quantizer into a node that is already "
                "affected by existing propagating quantizers (ids: {})!".format(
                    [pq.id for pq in target_node_affecting_quantizers]
                )
            )

        self._verify_nodes_and_edges_for_pq(prop_quantizer)

        for node_key in prop_quantizer.affected_operator_nodes:
            node = self.nodes[node_key]
            node_pqs = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            self._verify_qconfig_matching(prop_quantizer, node_pqs)
            node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        for node_key in prop_quantizer.affected_ip_nodes:
            node = self.nodes[node_key]
            node_pqs = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            self._verify_qconfig_matching(prop_quantizer, node_pqs)
            node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        for edge_key in prop_quantizer.affected_edges:
            edge = self.edges[edge_key]
            edge_pqs = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            self._verify_qconfig_matching(prop_quantizer, edge_pqs)
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer

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

        if cloned_prop_quant.unified_scale_type is not None:
            gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
            self._unified_scale_group_manager.add_to_group(gid, cloned_prop_quant)

        return cloned_prop_quant

    def remove_propagating_quantizer(
        self, prop_quantizer: PropagatingQuantizer, keep_propagating_quantizer_at_current_node=False
    ):
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
        if prop_quantizer.unified_scale_type is not None:
            gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(prop_quantizer.id)
            self._unified_scale_group_manager.remove_from_group(gid, prop_quantizer)
        self._pqs_after_weight_dependent_output_quantized_nodes.pop(prop_quantizer, None)

    def propagate_quantizer_via_path(
        self, prop_quantizer: PropagatingQuantizer, path: PropagationPath
    ) -> PropagatingQuantizer:
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
            if self.is_insertion_point(from_node_type):
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
                    if trait != QuantizationTrait.QUANTIZATION_AGNOSTIC:
                        target_node_list.append(successor_key)
                        return
                recursive_helper(successor_key, target_node_list)

        recursive_helper(node_key, ret_node_key_list)
        return ret_node_key_list

    def all_outputs_are_quantized(self, node_key) -> bool:
        """
        Returns True if all pathes from the given node to the first
        input quantable nodes have an activation quantizer, False otherwise.

        :param node_key: Given node key.
        :return: True if all pathes from the given node to the first
        input quantable nodes have an activation quantizer, False otherwise.
        """

        nodes_keys_stack = deque(self.successors(node_key))
        while nodes_keys_stack:
            node_key = nodes_keys_stack.popleft()
            node = self.nodes[node_key]
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait != QuantizationTrait.QUANTIZATION_AGNOSTIC:
                    return False
            elif node_type in [
                QuantizerPropagationStateGraphNodeType.PRE_HOOK,
                QuantizerPropagationStateGraphNodeType.POST_HOOK,
            ]:
                quantizer = node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if quantizer:
                    continue
            nodes_keys_stack.extend(self.successors(node_key))
        return True

    def get_paths_to_immediately_dominating_insertion_points(
        self, insertion_point_node_key: str
    ) -> List[PropagationPath]:
        group_dict = self.get_paths_to_immediately_dominating_insertion_points_grouped_by_unified_scales(
            insertion_point_node_key, set(), {}
        )
        return group_dict[None]

    def get_paths_to_immediately_dominating_insertion_points_grouped_by_unified_scales(
        self,
        insertion_point_node_key: str,
        unified_scale_op_metatypes: Set[Type[OperatorMetatype]],
        scales_unification_map: Dict[OperatorMetatype, OperatorMetatype],
    ) -> Dict[Optional[int], List[PropagationPath]]:
        """Paths are lists of edges."""
        next_group_idx = 0
        paths = {}

        def followed_by_weighted_types(curr_node_key, curr_node_metatype) -> bool:
            nodes_queue = deque(self.successors(curr_node_key))
            while nodes_queue:
                next_node_key = nodes_queue.popleft()
                next_node = self.nodes[next_node_key]
                next_node_type = next_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                if next_node_type != QuantizerPropagationStateGraphNodeType.OPERATOR:
                    nodes_queue.extend(self.successors(next_node_key))
                else:
                    next_node_metatype = next_node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                    next_node_trait = next_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                    if (
                        next_node_trait == QuantizationTrait.QUANTIZATION_AGNOSTIC
                        or next_node_metatype in unified_scale_op_metatypes
                    ):
                        nodes_queue.extend(self.successors(next_node_key))
                    if next_node_metatype in scales_unification_map[curr_node_metatype]:
                        return True
            return False

        def recursive_helper(curr_edge, curr_path, all_paths, curr_group):
            nonlocal next_group_idx
            curr_path.append(curr_edge)
            curr_node_key = curr_edge[0]
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(curr_node_type):
                if curr_group in all_paths:
                    all_paths[curr_group].append(curr_path)
                else:
                    all_paths[curr_group] = [curr_path]
                return

            if curr_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                metatype = curr_node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                unify_conditions = [
                    metatype in unified_scale_op_metatypes,
                    curr_group is None,
                    len(self.in_edges(curr_node_key)) > 1,
                ]
                if scales_unification_map is not None and metatype in scales_unification_map:
                    unify_conditions.append(followed_by_weighted_types(curr_node_key, metatype))
                if all(unify_conditions):
                    curr_group = next_group_idx
                    next_group_idx += 1

            for in_edge in self.in_edges(curr_node_key):
                path_copy = deepcopy(curr_path)
                recursive_helper(in_edge, path_copy, all_paths, curr_group)

        for in_edge in self.in_edges(insertion_point_node_key):
            if (
                self.nodes[in_edge[0]][QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                == QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER
            ):
                continue
            recursive_helper(in_edge, [], paths, curr_group=None)
        if not paths:
            paths[None] = []
        return paths

    def get_propagating_quantizers_immediately_dominated_by_node(self, node_key: str) -> Set[PropagatingQuantizer]:
        retval: Set[PropagatingQuantizer] = set()

        def traverse_fn(
            curr_node_key: str, all_pqs: Set[PropagatingQuantizer]
        ) -> Tuple[bool, Set[PropagatingQuantizer]]:
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(curr_node_type):
                pq = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if pq is not None:
                    all_pqs.add(pq)
                    return True, all_pqs
            return False, all_pqs

        self.traverse_graph(node_key, traverse_fn, retval)
        return retval

    def _build_branch_direct_output_dominators_info(self) -> Set[str]:
        """
        Traverses the graph backwards starting from outputs. If there is a path from an output to a branching node
        that only passes through quantization-agnostic ops, then this branching node is directly dominating an output.
        :return: The set of node names that directly dominate at least one output.
        """

        @dataclass
        class LocalState:
            global_result_ref: Set[str]
            encountered_quantizer_aware_ops: bool = False

        def traverse_fn(curr_node_key: str, local_state: LocalState) -> Tuple[bool, LocalState]:
            curr_node = self.nodes[curr_node_key]
            if len(list(self.successors(curr_node_key))) > 1:
                if not local_state.encountered_quantizer_aware_ops:
                    local_state.global_result_ref.add(curr_node_key)
                return True, local_state

            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if curr_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                node_trait = curr_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                op_meta = curr_node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                if op_meta not in OUTPUT_NOOP_METATYPES and node_trait in [
                    QuantizationTrait.INPUTS_QUANTIZABLE,
                    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
                    QuantizationTrait.NON_QUANTIZABLE,
                ]:
                    local_state.encountered_quantizer_aware_ops = True
            return False, local_state

        visited_node_keys = set()
        result = set()
        for output_node_key in self._output_node_keys_vs_nncf_nodes:
            output_state = LocalState(result)
            self._traverse_graph_recursive_helper(
                output_node_key, visited_node_keys, traverse_fn, output_state, traverse_backward=True, visit_once=False
            )
        return result

    def is_branching_node_dominating_outputs(self, from_node_key: str) -> bool:
        """
        Checks that all branches outgoing from the branching node can be quantized
        (They do not contain an output that should not be quantized).
        """
        if self._branch_nodes_directly_dominating_outputs is None:
            self._branch_nodes_directly_dominating_outputs = self._build_branch_direct_output_dominators_info()
        return from_node_key in self._branch_nodes_directly_dominating_outputs

    def get_visualized_graph(self):
        out_graph = nx.DiGraph()
        unified_scale_group_vs_pq_node_id_dict: Dict[int, List[str]] = {}
        for node_key, node in self.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(node_type):
                insertion_point_data: TargetPoint = node[
                    QuantizerPropagationStateGraph.QUANT_INSERTION_POINT_DATA_NODE_ATTR
                ]
                label = "TP: {}".format(str(insertion_point_data))
                out_graph.add_node(node_key, label=label, color="red")
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    prop_quantizer: PropagatingQuantizer = node[
                        QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR
                    ]
                    quant_node_key = "Quantizer #{}".format(prop_quantizer.id)
                    if prop_quantizer.potential_quant_configs:
                        quant_configs_str_list = [str(conf) for conf in prop_quantizer.potential_quant_configs]
                    else:
                        quant_configs_str_list = ["!!! NONE !!!]"]
                    sub_label = "[" + ",\n".join(quant_configs_str_list) + "]"
                    quant_node_label = quant_node_key + "\n" + "T: {}\n".format(sub_label)
                    quant_node_label += "Q-input sink ops: {}".format(
                        "\n".join(prop_quantizer.quantized_input_sink_operator_nodes)
                    )
                    pq_color = (
                        "blue"
                        if prop_quantizer not in self._pqs_after_weight_dependent_output_quantized_nodes
                        else "yellow"
                    )
                    out_graph.add_node(quant_node_key, color=pq_color, label=quant_node_label)
                    out_graph.add_edge(quant_node_key, node_key, style="dashed")
                    if prop_quantizer.unified_scale_type is not None:
                        gid = self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(
                            prop_quantizer.id
                        )
                        if gid in unified_scale_group_vs_pq_node_id_dict:
                            unified_scale_group_vs_pq_node_id_dict[gid].append(quant_node_key)
                        else:
                            unified_scale_group_vs_pq_node_id_dict[gid] = [quant_node_key]

            elif node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                out_graph.add_node(node_key)
            elif node_type == QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER:
                out_graph.add_node(node_key, color="green", label=node["label"])
            else:
                raise nncf.InternalError("Invalid QuantizerPropagationStateGraph node!")
        for u, v in self.edges:
            edge = self.edges[u, v]
            attrs = {}
            affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if affecting_quantizers:
                label = ", ".join([str(pq.id) for pq in affecting_quantizers])
                attrs = {"color": "blue", "label": label}
            is_integer_path = edge[QuantizerPropagationStateGraph.IS_INTEGER_PATH_EDGE_ATTR]
            if is_integer_path:
                attrs = {"color": "violet", "style": "bold"}
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
                out_graph.add_edge(
                    curr_pq_node_key,
                    next_pq_node_key,
                    arrowhead="none",
                    style="dotted",
                    label="Unified group {}".format(gid),
                )

        return out_graph

    def traverse_graph(
        self,
        curr_node_key: str,
        traverse_function: Callable[[str, Any], Tuple[bool, Any]],
        output: Any,
        traverse_forward: bool = True,
        dfs: bool = True,
    ) -> Any:
        visited_node_keys: Set[str] = set()
        node_keys_to_visit: Deque[Tuple[str, Any]] = deque()
        next_node_keys_indexer = self.succ if traverse_forward else self.pred
        # Storing the node-specific operation output is required so that this function
        # interface could generalize to situations where 'output' is not a global storage
        # for some sort of data to be gathered from the graph as a whole, but is a traversal history-
        # aware node-specific output, such as which quantizer affects the current node.
        node_keys_to_visit.appendleft((curr_node_key, output))

        while node_keys_to_visit:
            if dfs:
                node_key, local_output = node_keys_to_visit.popleft()
            else:
                node_key, local_output = node_keys_to_visit.pop()
            is_finished, new_output = traverse_function(node_key, local_output)
            visited_node_keys.add(node_key)
            if not is_finished:
                for next_node_key in next_node_keys_indexer[node_key]:
                    if next_node_key not in visited_node_keys:
                        node_keys_to_visit.appendleft((next_node_key, new_output))

        return output

    def _traverse_graph_recursive_helper(
        self,
        curr_node_key: str,
        visited_node_keys: Set[str],
        traverse_function: Callable[[str, Any], Tuple[bool, Any]],
        output: Any,
        traverse_backward: bool = False,
        visit_once: bool = True,
    ):
        """This is DFS, and may fail with 'maximum recursion depth exceeded' for complex graphs."""
        is_finished, output = traverse_function(curr_node_key, output)
        if visit_once:
            visited_node_keys.add(curr_node_key)
        next_node_keys_indexer = self.pred if traverse_backward else self.succ
        if not is_finished:
            for node_key in next_node_keys_indexer[curr_node_key]:
                if visit_once and node_key in visited_node_keys:
                    continue
                self._traverse_graph_recursive_helper(
                    node_key, visited_node_keys, traverse_function, output, traverse_backward, visit_once
                )
        return output

    def _get_next_prop_quantizer_id(self):
        self._created_prop_quantizer_counter += 1
        return self._created_prop_quantizer_counter

    def _is_position_accepting(self, ip_node_key: str):
        return True

    def get_unified_scale_group_id_by_propagating_quantizer_id(self, pqid: int) -> int:
        return self._unified_scale_group_manager.get_group_id_by_propagating_quantizer_id(pqid)

    def get_quantizers_at_input_nncf_nodes(self) -> Dict[NNCFNode, List[int]]:
        retval: Dict[NNCFNode, List[int]] = {}

        def recursive_helper(curr_node_key: str, curr_input_quantizer_ids_list: List[int]):
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]

            if self.is_insertion_point(curr_node_type):
                pq = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
                if pq is not None:
                    curr_input_quantizer_ids_list.append(pq.id)
                    return
            elif curr_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = curr_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait != QuantizationTrait.QUANTIZATION_AGNOSTIC:
                    return
            elif curr_node_type == QuantizerPropagationStateGraphNodeType.AUXILIARY_BARRIER:
                return

            for successor_key in self.successors(curr_node_key):
                recursive_helper(successor_key, curr_input_quantizer_ids_list)

        for input_node_key, input_nncf_node in self._input_node_keys_vs_nncf_nodes.items():
            current_input_quantizer_ids = []
            recursive_helper(input_node_key, current_input_quantizer_ids)
            retval[input_nncf_node] = current_input_quantizer_ids

        return retval

    def merge_redundant_subsequent_quantizers_across_graph(self):
        def is_downstream_quantizer_redundant(
            downstream_quantizer: PropagatingQuantizer, upstream_quantizer: PropagatingQuantizer
        ):
            ds_configs = downstream_quantizer.potential_quant_configs
            us_configs = upstream_quantizer.potential_quant_configs
            assert len(ds_configs) == 1
            assert len(us_configs) == 1
            ds_config = ds_configs[0]
            us_config = us_configs[0]
            is_redundant = True
            is_redundant = is_redundant and (ds_config.num_bits == us_config.num_bits)

            # Avoid asymmetric quantization if a symmetrically quantized tensor arrived
            is_redundant = is_redundant and (
                (ds_config.mode == us_config.mode)
                or (ds_config.mode == QuantizationMode.ASYMMETRIC and us_config.mode == QuantizationMode.SYMMETRIC)
            )

            # Avoid per-channel quantization if a per-tensor-quantized tensor arrived
            is_redundant = is_redundant and (
                (ds_config.per_channel == us_config.per_channel)
                or (ds_config.per_channel is True and us_config.per_channel is False)
            )
            return is_redundant

        def merge_traverse_fn(
            curr_node_key: str, affecting_pq_and_prev_node_key: Tuple[Optional[PropagatingQuantizer], str]
        ) -> Tuple[Optional[PropagatingQuantizer], str]:
            # For this to work, DFS must be used for graph traversal. Also, this only
            # works with the generic traverse_graph interface because of
            # Python's pass-by-value mechanism for tuples.
            affecting_pq, prev_node_key = affecting_pq_and_prev_node_key
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]

            # Skipping traversing through the INTEGER path.
            if curr_node_key != prev_node_key:
                edge = self.edges[prev_node_key, curr_node_key]
                if edge[QuantizerPropagationStateGraph.IS_INTEGER_PATH_EDGE_ATTR]:
                    return False, (None, curr_node_key)

            if self.is_insertion_point(curr_node_type):
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
        retval: Set[PropagatingQuantizer] = set()

        def traverse_fn(
            curr_node_key: str, all_pqs: Set[PropagatingQuantizer]
        ) -> Tuple[bool, Set[PropagatingQuantizer]]:
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(curr_node_type):
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

    def get_quant_insertion_point_for_propagating_quantizer(
        self, prop_quant: PropagatingQuantizer
    ) -> QuantizationInsertionPointBase:
        final_node_key = prop_quant.current_location_node_key
        final_node = self.nodes[final_node_key]
        insertion_point = final_node[QuantizerPropagationStateGraph.QUANT_INSERTION_POINT_DATA_NODE_ATTR]
        return insertion_point

    def _get_all_quantizers_grouped_by_affecting_op_set(self) -> List[SharedAffectedOpsPropagatingQuantizerGroup]:
        all_pqs = self.collect_all_propagating_quantizers()

        class Grouper:
            """
            Propagating quantizers will be grouped so that each quantizer is in the same group as the
            node that it is affecting. Furthermore, each quantizer that does not affect any node
            (e.g. if it only affects other quantizers as a topmost quantizer in a requantization
            scenario) will be placed in a separate group.
            """

            def __init__(self):
                self._group_vs_node_keys_and_pqs: Dict[int, SharedAffectedOpsPropagatingQuantizerGroup] = {}
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
                self._group_vs_node_keys_and_pqs[new_gid] = SharedAffectedOpsPropagatingQuantizerGroup(
                    {pq}, set(pq.quantized_input_sink_operator_nodes)
                )
                new_group_data = self._group_vs_node_keys_and_pqs[new_gid]
                gids_to_merge: Set[int] = set()
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
        assert (
            self.nodes[operator_node_key][QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            == QuantizerPropagationStateGraphNodeType.OPERATOR
        )
        return len(list(self.predecessors(operator_node_key)))

    def create_quantizer_setup(
        self, weight_quantizable_node_names_vs_configs: Dict[NNCFNodeName, List[QuantizerConfig]]
    ) -> MultiConfigQuantizerSetup:
        same_op_groups = self._get_all_quantizers_grouped_by_affecting_op_set()
        setup = MultiConfigQuantizerSetup()

        pqid_vs_qpid: Dict[int, QuantizationPointId] = {}
        qm_node_vs_same_op_gid: Dict[NNCFNodeName, int] = {}
        for group in same_op_groups:
            grouped_ids = set()
            for pq in group.affecting_prop_quants:
                directly_quantized_operator_node_names = [
                    next(iter(self.op_node_keys_to_underlying_nodes_mapping[key])).node_name
                    for key in pq.quantized_input_sink_operator_nodes
                ]
                if pq.downstream_propagating_quantizers:
                    affected_operator_nodes = set()
                    for apq in pq.downstream_propagating_quantizers:
                        affected_operator_nodes.update(apq.quantized_input_sink_operator_nodes)
                    directly_quantized_operator_node_names = [
                        next(iter(self.op_node_keys_to_underlying_nodes_mapping[key])).node_name
                        for key in pq.quantized_input_sink_operator_nodes - affected_operator_nodes
                    ]
                quant_point = MultiConfigQuantizationPoint(
                    self.get_quant_insertion_point_for_propagating_quantizer(pq),
                    pq.potential_quant_configs,
                    directly_quantized_operator_node_names,
                )
                qp_id = pq.id
                pqid_vs_qpid[pq.id] = qp_id
                setup.quantization_points[qp_id] = quant_point
                grouped_ids.add(qp_id)

            gid = setup.register_shared_inputs_group(list(grouped_ids))
            for weighted_node_name in weight_quantizable_node_names_vs_configs:
                for affected_node_key in group.affected_op_node_keys:
                    underlying_node_names = [
                        n.node_name for n in self.op_node_keys_to_underlying_nodes_mapping[affected_node_key]
                    ]
                    if weighted_node_name in underlying_node_names:
                        qm_node_vs_same_op_gid[weighted_node_name] = gid

        if setup.quantization_points.keys():
            max_aq_id = max(setup.quantization_points.keys()) + 1
        else:
            max_aq_id = 0

        next_wq_id = max_aq_id + 1
        wao_op_node_key_vs_wq_id: Dict[str, QuantizationPointId] = {}
        for weighted_node_name, qconfig_list in weight_quantizable_node_names_vs_configs.items():
            quant_point = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(weighted_node_name), qconfig_list, [weighted_node_name]
            )
            setup.quantization_points[next_wq_id] = quant_point
            if weighted_node_name not in qm_node_vs_same_op_gid:
                # Happens for LSTM cells. The "hidden" Linear layer, as represented in NNCFGraph, has no
                # input edges, since its input is not a regular network input, but a recurrent input
                # from the previous execution step. TODO: extend recurrent operations handling so that NNCF graph
                # has information on which operation accepts recurrent inputs.
                nncf_logger.debug(
                    "Could not find an associated input activation quantizer "
                    "for a weighted node with quantizable weights: {}\n".format(weighted_node_name)
                )
            else:
                associated_same_op_gid = qm_node_vs_same_op_gid[weighted_node_name]
                setup.shared_input_operation_set_groups[associated_same_op_gid].add(next_wq_id)

            for wao_op_node_key in self._pqs_after_weight_dependent_output_quantized_nodes.values():
                underlying_node_names = [
                    n.node_name for n in self.op_node_keys_to_underlying_nodes_mapping[wao_op_node_key]
                ]
                if weighted_node_name in underlying_node_names:
                    wao_op_node_key_vs_wq_id[wao_op_node_key] = next_wq_id
            next_wq_id += 1

        pq_sets_grouped_by_unified_scale = list(
            self._unified_scale_group_manager.get_group_vs_prop_quants_dict().values()
        )
        for pq_set in pq_sets_grouped_by_unified_scale:
            setup.register_unified_scale_group_with_types(
                [pqid_vs_qpid[pq.id] for pq in pq_set], [pq.unified_scale_type for pq in pq_set]
            )

        setup = self._handle_output_quantizers_for_weights_as_outputs_ops(setup, pqid_vs_qpid, wao_op_node_key_vs_wq_id)

        return setup

    def _handle_output_quantizers_for_weights_as_outputs_ops(
        self,
        setup: MultiConfigQuantizerSetup,
        pqid_vs_qpid: Dict[int, QuantizationPointId],
        wao_op_node_key_vs_wq_id: Dict[str, QuantizationPointId],
    ) -> MultiConfigQuantizerSetup:
        """
        In case there are propagating quantizers dependent on the weights-as-outputs weighted operations
        (as marked by mark_act_quantizer_as_dependent_on_weights) in the current state of the quantizer setup,
        and if the quantizer configurations between the dependent activation quantizer and the weight output
        quantizer have at least one compatible configuration (checked across all AQ's in the unified
        scale group of the dependent AQ), then the activation quantizer will be removed and the weight quantizer's
        config options will be limited to the common configurations between the dependent quantizer and the
        original weight quantizer configuration space. In case the dependent quantizer to be removed
        belonged to a unified scale group, the weight quantizer will be put into the same group instead.
        If the configurations were incompatible, will not remove the corresponding activation quantizer and
        requantization will occur.

        :param: setup - a MultiConfigQuantizerSetup corresponding to the quantizer setup state with potentially
            dependent activation quantizers on the weights-as-outputs ops
        :param: pqid_vs_qpid - a mapping from propagating quantizer IDs to the corresponding activation quantization
            point IDs in `setup`
        :param: wao_op_node_key_vs_wq_id - a mapping from weights-as-outputs operator node keys in the
            QuantizerPropagationStageGraph to the corresponding weight quantization points in `setup`
        :return: A MultiConfigQuantizerSetup with weights-as-outputs-dependent quantizers removed where possible
            and shared inputs/unified scales group adjusted to reflect the change.
        """

        # For the weights-are-outputs quantized operations, need to find out the dependent activation quantizers in
        # the multiconfig setup and see if it is possible to avoid requantization by selecting a common configuration
        # subset. If yes and the activation quantizer becomes unnecessary, need to unify the scales of the weight
        # quantizer if the removed activation quantizer also had unified scales. If requantization is unavoidable,
        # leave quantizers as-is (do not unify weight quantizer scales).
        for pq, wao_op_node_key in self._pqs_after_weight_dependent_output_quantized_nodes.items():
            wao_qp_id = wao_op_node_key_vs_wq_id[wao_op_node_key]
            curr_intersection_of_qconfigs = setup.quantization_points[wao_qp_id].possible_qconfigs
            qp_id_for_current_pq = pqid_vs_qpid[pq.id]

            # Irrespective of whether the dependent input activation quantizer gets merged into
            # the weight quantizer, need to register the weight quantizer into the same shared input
            # group as the dependent input activation quantizer.
            shared_input_gid = setup.get_shared_inputs_group_id(qp_id_for_current_pq)
            if shared_input_gid is not None:
                setup.register_existing_qp_id_in_shared_input_group(wao_qp_id, shared_input_gid)

            unified_scale_gid = setup.get_unified_scale_group_id(qp_id_for_current_pq)
            if unified_scale_gid is not None:
                all_qp_ids_in_unified_scale_group = deepcopy(setup.unified_scale_groups[unified_scale_gid])
            else:
                all_qp_ids_in_unified_scale_group = {qp_id_for_current_pq}
            for act_qp_id in all_qp_ids_in_unified_scale_group:
                curr_act_qconfigs = setup.quantization_points[act_qp_id].possible_qconfigs
                curr_intersection_of_qconfigs = self._get_weight_and_activation_qconfig_list_intersection(
                    curr_intersection_of_qconfigs, curr_act_qconfigs
                )

            # Do further filtering for per-tensor quantizations only.
            # TODO: relax the requirement to allow the scale shape of the weight-as-output quantizer
            # matching the scale shape of the output quantizer (which may, in theory, end up being per-channel
            curr_intersection_of_qconfigs = list(filter(lambda x: not x.per_channel, curr_intersection_of_qconfigs))

            if not curr_intersection_of_qconfigs:
                # Requantization is unavoidable
                nncf_logger.debug(
                    f"Attempted to use weight quantizer of {wao_op_node_key} "
                    f"to quantize input of {pq.affected_operator_nodes}, "
                    f"but no compatible configs were found."
                )
                continue

            setup.quantization_points[wao_qp_id].possible_qconfigs = curr_intersection_of_qconfigs
            for act_qp_id in all_qp_ids_in_unified_scale_group:
                setup.quantization_points[act_qp_id].possible_qconfigs = curr_intersection_of_qconfigs

            if unified_scale_gid is not None:
                setup.register_existing_qp_id_in_unified_scale_group(wao_qp_id, unified_scale_gid)
                unified_scale_qp_printable_str = ", ".join(
                    [str(setup.quantization_points[qp_id]) for qp_id in all_qp_ids_in_unified_scale_group]
                )
                nncf_logger.debug(
                    f"Unifying weight quantizer ranges of {wao_op_node_key} with {unified_scale_qp_printable_str}"
                )

            # The activation quantizer is now unnecessary since we could find a matching weight quantization
            # for the op. Should discard it, but first transfer the knowledge on the operators it quantizes downstream
            # to the weights-as-outputs quantization point.
            dir_quant_ops = setup.quantization_points[qp_id_for_current_pq].directly_quantized_operator_node_names
            setup.quantization_points[wao_qp_id].directly_quantized_operator_node_names.extend(deepcopy(dir_quant_ops))
            setup.discard(qp_id_for_current_pq, keep_shared_input_qps=True)
        return setup

    @staticmethod
    def _get_weight_and_activation_qconfig_list_intersection(
        weight_qconfig_options: List[QuantizerConfig], activation_qconfig_options: List[QuantizerConfig]
    ) -> List[QuantizerConfig]:
        """
        Returns special intersection between weight and activation quantization configurations.

        :param weight_qconfig_options: List of QuantizerConfig associated with weights.
        :param activation_qconfig_options: List of QuantizerConfig associated with activations.
        :return: Special intersection between configurations.
        """
        act_qconfig_extend_list = []
        for act_qconfig in activation_qconfig_options:
            if act_qconfig.signedness_to_force is None:
                for signedness_to_force_position in [True, False]:
                    act_qconfig_updated = deepcopy(act_qconfig)
                    act_qconfig_updated.signedness_to_force = signedness_to_force_position
                    act_qconfig_extend_list.append(act_qconfig_updated)
        act_qconfig_extend_list += activation_qconfig_options
        return [qconf for qconf in weight_qconfig_options if qconf in act_qconfig_extend_list]

    def run_consistency_check(self) -> bool:
        all_pqs = self.collect_all_propagating_quantizers()

        def traverse_fn(curr_node_key: str, unused) -> Tuple[bool, Any]:
            nncf_logger.debug(f"Processing node: {curr_node_key}")
            node = self.nodes[curr_node_key]
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if self.is_insertion_point(node_type):
                pq: PropagatingQuantizer = node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
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
                nncf_logger.debug(f"Processing edge: {out_edge_key}")
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
                assert self.is_insertion_point(ip_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR])
                assert pq in ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            for affected_op_node_key in pq.affected_operator_nodes:
                op_node = self.nodes[affected_op_node_key]
                assert (
                    op_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                    == QuantizerPropagationStateGraphNodeType.OPERATOR
                )
                assert pq in op_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
