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
from collections import OrderedDict
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Deque, Dict, List, Optional, Set, Tuple

import networkx as nx

import nncf
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.hardware.config import HWConfig
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph
from nncf.common.quantization.quantizer_propagation.grouping import QuantizersWaitingForMergeManager
from nncf.common.quantization.quantizer_propagation.structs import IgnoreReason
from nncf.common.quantization.quantizer_propagation.structs import PropagatingQuantizer
from nncf.common.quantization.quantizer_propagation.structs import PropagationPath
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationRule
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationStateGraphNodeType
from nncf.common.quantization.quantizer_setup import DEFAULT_QUANTIZER_CONFIG
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import UnifiedScaleType
from nncf.common.scopes import matches_any
from nncf.common.scopes import should_consider_scope
from nncf.common.utils.debug import DEBUG_LOG_DIR
from nncf.common.utils.debug import is_debug
from nncf.common.utils.dot_file_rw import write_dot_graph


class TransitionStatus(Enum):
    SHOULD_TRANSITION = 0
    SHOULD_MERGE = 1
    SHOULD_NOT_TRANSITION = 2
    SHOULD_WAIT_FOR_MERGE = 3


class FinalizedQuantizationProposal:
    """
    Describes a version of QuantizationProposal in which a single quantizer configuration has been chosen
    (using one or the other way of disambiguation) for each quantization point in the setup that was made available
    in the original QuantizationProposal
    """

    def __init__(
        self,
        single_config_quantizer_setup: SingleConfigQuantizerSetup,
        quant_prop_graph: QuantizerPropagationStateGraph,
    ):
        """
        :param single_config_quantizer_setup: The single-configuration quantizer setup.
        :param quant_prop_graph: The quantizer propagation state graph to which this quantizer setup is related.
        """
        self.single_config_quantizer_setup = single_config_quantizer_setup
        self._quant_prop_graph = quant_prop_graph

    @property
    def quant_prop_graph(self):
        return self._quant_prop_graph


class QuantizationProposal:
    """
    Describes an intermediate state in the quantizer setup creation, at which the quantizers have already been
    propagated until a standstill, and each quantizer still has more than one (in general) quantizer configurations
    to be chosen from. This object serves as an input to the external algorithm (such as HAWQ or AutoQ) that
    disambiguates the quantizer configurations at each location without changing the quantizer positions.
    """

    def __init__(
        self,
        quantizer_setup: MultiConfigQuantizerSetup,
        quant_prop_graph: QuantizerPropagationStateGraph,
        quantization_point_id_vs_prop_quantizer: Dict[QuantizationPointId, PropagatingQuantizer],
    ):
        """
        :param quantizer_setup: The MultiConfigQuantizerSetup object obtained from a quantizer propagation solver.
        :param quant_prop_graph: The QuantizerPropagationStateGraph whose state corresponds to the `quantizer_setup`,
          also obtained from the solver
        :param quantization_point_id_vs_prop_quantizer: A mapping of the quantization point IDs in `quantizer_setup` to
          propagating quantizers registered in `quant_prop_graph`.
        """
        self.quantizer_setup = quantizer_setup
        self._quant_prop_graph = quant_prop_graph
        self._quantization_point_id_vs_prop_quantizer = quantization_point_id_vs_prop_quantizer
        self._prop_quantizer_vs_quantization_point_id: Dict[PropagatingQuantizer, QuantizationPointId] = {}
        for qp_id, pq in self._quantization_point_id_vs_prop_quantizer.items():
            self._prop_quantizer_vs_quantization_point_id[pq] = qp_id

    def constrain_quantizer_config_list_for_insertion(
        self, quantization_point_id: QuantizationPointId, constrained_config_list: List[QuantizerConfig]
    ):
        """
        Constrains a set of available quantizer configurations for a quantization point with a given ID as
        defined by the list of quantizer configurations - in essence, performs a selection.

        :param quantization_point_id: The ID of the quantization point.
        :param constrained_config_list: The list of configs (of which every config is already present in the
          currently available in the quantization point's set of available config) that will replace the list
          of the quantizer configs for the quantization point defined by `quantization_point_id`.
        """
        prior_list = self.quantizer_setup.quantization_points[quantization_point_id].possible_qconfigs
        if not all(qc in prior_list for qc in constrained_config_list):
            raise nncf.InternalError(
                "Constrained config list is incompatible with the result of the quantizer propagation!"
            )
        # TODO (vshampor): only allow to constrain 'input-group'-wise?
        self.quantizer_setup.quantization_points[quantization_point_id].possible_qconfigs = constrained_config_list

        if quantization_point_id in self._quantization_point_id_vs_prop_quantizer:
            pq = self._quantization_point_id_vs_prop_quantizer[quantization_point_id]
            pq.potential_quant_configs = constrained_config_list

    def finalize(self, final_quantizer_setup: SingleConfigQuantizerSetup, strict=True) -> FinalizedQuantizationProposal:
        """
        Given a single-configuration quantizer setup (which is constructed by picking a single quantizer configuration
        for each of the multi-configuration quantization points in this proposal's multi-config setup), prepares a
        finalized proposal ready to be turned into a final single-config quantizer setup by the solver.
        :param final_quantizer_setup: The single-configuration quantizer setup, disambiguated from this proposal's
          multi-config setup.
        :param strict:
        :return: If False, will allow quantizer configurations in the `final_quantizer_setup` that were not present
        at the same quantization point in this proposal's multi-config quantizer setup.
        """
        for pq, qp_id in self._prop_quantizer_vs_quantization_point_id.items():
            if qp_id not in final_quantizer_setup.quantization_points:
                self._quant_prop_graph.remove_propagating_quantizer(pq)
            else:
                final_qconfig = final_quantizer_setup.quantization_points[qp_id].qconfig
                if strict:

                    def is_final_qconfig_compatible_to_initial(initial_qconfig: QuantizerConfig):
                        return (
                            final_qconfig.per_channel == initial_qconfig.per_channel
                            and final_qconfig.mode == initial_qconfig.mode
                            and final_qconfig.num_bits == initial_qconfig.num_bits
                            and (
                                final_qconfig.signedness_to_force == initial_qconfig.signedness_to_force
                                or initial_qconfig.signedness_to_force is None
                                or final_qconfig.signedness_to_force is None
                            )
                        )

                    compatible_initial_qconfs = list(
                        filter(
                            is_final_qconfig_compatible_to_initial,
                            self.quantizer_setup.quantization_points[qp_id].possible_qconfigs,
                        )
                    )
                    if not compatible_initial_qconfs:
                        raise nncf.InternalError(
                            "The final quantizer setup has configurations that were not present in the "
                            "initial proposal!"
                        )
                    if final_qconfig.signedness_to_force is None:
                        initial_qconfs_signedness_values = {qc.signedness_to_force for qc in compatible_initial_qconfs}
                        if None not in initial_qconfs_signedness_values and len(initial_qconfs_signedness_values) == 1:
                            # The initial configs were either all forced-signed or all forced-unsigned - should set
                            # final qconfig's forced field appropriately
                            final_qconfig.signedness_to_force = initial_qconfs_signedness_values.pop()

                pq.potential_quant_configs = [final_qconfig]
        return FinalizedQuantizationProposal(final_quantizer_setup, self._quant_prop_graph)


class PostprocessingNodeLocator:
    """
    Detects the nodes in the QuantizerPropagationStateGraph, which implement the post-processing logic in the model.
    Based on the special post-processing marker metatypes the nodes are placed in the ignored.
    """

    def __init__(
        self,
        quant_prop_graph: QuantizerPropagationStateGraph,
        quantizable_layer_nodes: List[QuantizableWeightedLayerNode],
        post_processing_marker_metatypes: List[OperatorMetatype],
    ):
        self._quant_prop_graph = quant_prop_graph
        self._post_processing_marker_metatypes = post_processing_marker_metatypes
        self._quantizable_layer_node_keys = [q_nodes.node.node_key for q_nodes in quantizable_layer_nodes]
        self._post_processing_marker_encountered = False

    def _is_node_has_underlying_weights(self, node_key: str) -> bool:
        if not self._is_node_operator(node_key):
            return False
        underlying_nncf_nodes = self._quant_prop_graph.op_node_keys_to_underlying_nodes_mapping[node_key]
        for node in underlying_nncf_nodes:
            if node.node_key in self._quantizable_layer_node_keys:
                return True
        return False

    def _get_node_metatype(self, node_key: str) -> OperatorMetatype:
        node = self._quant_prop_graph.nodes[node_key]
        return node.get(self._quant_prop_graph.OPERATOR_METATYPE_NODE_ATTR)

    def _is_node_operator(self, node_key: str) -> bool:
        node = self._quant_prop_graph.nodes[node_key]
        return node.get(self._quant_prop_graph.NODE_TYPE_NODE_ATTR) == QuantizerPropagationStateGraphNodeType.OPERATOR

    def get_post_processing_node_keys(self) -> Set[str]:
        """
        Finds out the nodes of the QuantizerPropagationStateGraph, which are in post-processing part of the model.
        Starting from the output nodes all the nodes are added to path,
        until the quantizable nodes with weights are faced.
        If the path with the nodes has the post-processing marker node,
        all the nodes in this path (except outputs and nodes with weights) will be added into ignored.

        :return: Set of the node keys to be ignored.
        """
        output_nodes = []
        for output_metatype in OUTPUT_NOOP_METATYPES.values():
            output_nodes.extend(self._quant_prop_graph.get_node_keys_by_metatype(output_metatype))

        def get_ignored_operations(output_nodes: List[str]) -> Tuple[Set[str], Set[str]]:
            stack = [([start_node_key], False) for start_node_key in output_nodes]
            ignored_operations = set()

            def _extend_ignored_operations(path: List[str]):
                for node in path:
                    if (
                        self._is_node_operator(node)
                        and not self._is_node_has_underlying_weights(node)
                        and node not in output_nodes
                    ):
                        ignored_operations.add(node)

            visited = set()
            while stack:
                path, post_proc_encountered = stack.pop()
                node_key = path[-1]
                visited.add(node_key)
                if (
                    self._is_node_operator(node_key)
                    and self._get_node_metatype(node_key) in self._post_processing_marker_metatypes
                ):
                    post_proc_encountered = True

                if (
                    self._is_node_has_underlying_weights(node_key)
                    or node_key in self._quant_prop_graph.get_input_node_keys()
                ):
                    if post_proc_encountered:
                        _extend_ignored_operations(path)
                else:
                    for input_key in self._quant_prop_graph.predecessors(node_key):
                        if input_key in visited and post_proc_encountered and input_key in ignored_operations:
                            # We have already visited input node, encountered post_processing node in current path,
                            # and marked input node as ignored, then we can add entire path to ignored_operations
                            _extend_ignored_operations(path)
                        elif input_key in visited and not post_proc_encountered and input_key not in ignored_operations:
                            # We have already visited input node
                            # but did not add it to ignored_operations (no post_processing node above)
                            # and did not encounter post_processing node in current path,
                            # then we can stop traversal
                            pass
                        else:
                            stack.append((path + [input_key], post_proc_encountered))
            return ignored_operations

        ignored_ops = get_ignored_operations(output_nodes)
        return ignored_ops


class QuantizerPropagationSolver:
    """
    Analyzes a fresh QuantizerPropagationStateGraph object according to HW
    configuration supplied in the initializer and produces the list of insertion
    commands that correspond to the final state of the quantizer propagation graph
    when the model has the most control flow graph edges quantized according to HW
    capabilities.
    """

    DEFAULT_QUANTIZATION_TYPES = [
        QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False)
    ]

    DEFAULT_PROPAGATION_STRATEGY = QuantizerPropagationRule.MERGE_ALL_IN_ONE

    def __init__(
        self,
        activation_ignored_scopes: Dict[str, IgnoreReason] = None,
        weight_ignored_scopes: List[str] = None,
        activation_target_scopes: List[str] = None,
        weight_target_scopes: List[str] = None,
        hw_config: HWConfig = None,
        default_trait_to_metatype_map: Dict[QuantizationTrait, List[OperatorMetatype]] = None,
        propagation_strategy: QuantizerPropagationRule = None,
        default_qconfig_list: List[QuantizerConfig] = None,
        quantizable_layer_nodes: List[QuantizableWeightedLayerNode] = None,
        scope_overrides: Dict = None,
        global_constraints: Dict[QuantizerGroup, QuantizationConstraints] = None,
        additional_unified_scale_op_scopes: List[List[str]] = None,
        run_consistency_checks: bool = False,
        quantize_outputs: bool = False,
        post_processing_marker_metatypes: List[OperatorMetatype] = None,
        metatypes_to_ignore: List[OperatorMetatype] = None,
        scales_unification_map: Dict[OperatorMetatype, OperatorMetatype] = None,
    ):
        """
        Initializes the solver with parameters affecting the resulting quantizer setup.

        :param activation_ignored_scopes: A dict with key as node name and value as ignore reason
          to match against NNCFGraph node names and ignore matching nodes.
          Ignored nodes will not have quantizers applied to their activation inputs
          (even if required by node's metatype and HW config), and the downstream quantizers will not propagate
          upwards through the corresponding node.
        :param weight_ignored_scopes: A list of strings to match against NNCFGraph node names
          and ignore matching nodes. Ignored nodes will not have quantizers applied to their weight inputs
          (even if required by node's metatype and HW config).
        :param activation_target_scopes: A list of strings to match against NNCFGraph and define a set of nodes
          to be considered during quantizer propagation. When `activation_ignored_scopes` is a "denylist",
          the `activation_target_scopes` is an "allowlist";
          otherwise, same effects apply as for `activation_ignored_scopes`.
        :param weight_target_scopes: A list of strings to match against NNCFGraph and define a set of nodes
          which weights should be quantized. When `weight_ignored_scopes` is a "denylist",
          the `weight_target_scopes` is an "allowlist"; otherwise, same effects apply as for `weight_ignored_scopes`.
        :param hw_config: A hardware config to be used for determining the set of operations to be quantized with
        respect to their inputs and, for every such operation, the set of allowed quantizer configurations
        :param default_trait_to_metatype_map: The mapping of QuantizationTrait's to the metatypes to be associated with
        these by default. Used if no HW config is passed, or if an operation that is unknown to HW config is
        encountered.
        :param propagation_strategy: The strategy to be used while propagating and merging quantizers.
        :param default_qconfig_list: The list of quantizer configurations that should be applied for quantizing
          inputs of operations for which the `hw_config` has no explicit or implicit information on how to
          quantize them.
        :param quantizable_layer_nodes: A list of NNCFGraph node names that correspond to operations with
          quantizable weights, along with the corresponding allowed quantizer configurations. Required to
          build a complete quantizer setup and impacts the activation quantizer propagation in certain
          cases.
        :param scope_overrides: A dictionary of quantization configuration overrides for inputs to matching
          operation nodes.
        :param global_constraints: Global quantizer configuration constraints that will be applied to
          what is specified in the HW config to limit the initial set of possible quantizer configurations
          for input of each operation.
        :param additional_unified_scale_op_scopes: A list of strings to match against NNCFGraph node names,
          inputs of which must be always quantized with a single scale, i.e. with a single set of
          trainable quantizer parameters.
        :param run_consistency_checks: Whether to run internal consistency checks at each propagation step.
        :param quantize_outputs: Whether to insert additional quantizers right before each of the model outputs.
        :param post_processing_marker_metatypes: The framework specific NNCF metatypes, which are markers for
            the model post-processing part. They are used for automatic ignoring post-processing nodes.
            The seeking post-processing nodes algorithm uses traversing through the model graph from the output nodes.
            During traversing all the visited nodes are added, until the quantizable nodes with weights are faced.
            If the path with the nodes has the post-processing marker node,
            all the nodes in this path will be added into ignored.
            If None automatic ignoring will be skipped.
        :param metatypes_to_ignore: The framework specific NNCF metatypes,
            which should be automatically ignored.
        :param scales_unification_map: The framework-specific map with NNCF metatypes, which generating a quantizer
            that can be unified if it so requires based on metatype.
        """
        if default_trait_to_metatype_map is None:
            self._default_trait_to_metatype_map = {}
        else:
            self._default_trait_to_metatype_map = default_trait_to_metatype_map
        self.default_global_qconfig_list = default_qconfig_list
        self._hw_config: HWConfig = hw_config
        self._visualizer = None
        if is_debug():
            from nncf.common.quantization.quantizer_propagation.visualizer import QuantizerPropagationVisualizer

            self._visualizer = QuantizerPropagationVisualizer(DEBUG_LOG_DIR + "/quant_prop")
        self._propagation_strategy = (
            propagation_strategy if propagation_strategy else QuantizerPropagationSolver.DEFAULT_PROPAGATION_STRATEGY
        )  # TODO (vshampor): determine from config
        self._operator_quantization_trait_map = self.get_operator_quantization_traits_map()
        self._operator_allowed_qconfigs_map = self._get_operator_qconfigs_map()
        self._quantize_outputs = quantize_outputs
        self._ignored_scopes = activation_ignored_scopes
        self._target_scopes = activation_target_scopes
        self._weight_quantizable_node_names_vs_qconfigs = self._filter_by_weight_ignored_target_scopes(
            quantizable_layer_nodes, weight_ignored_scopes, weight_target_scopes
        )

        if scope_overrides is None:
            self._scope_overrides = {}
        else:
            self._scope_overrides: Dict = scope_overrides
        self._global_constraints: Dict["QuantizerGroup", "QuantizationConstraints"] = global_constraints
        self._run_consistency_checks = run_consistency_checks

        self._unified_scales_operation_set = set()
        if self._hw_config is not None:
            self._unified_scales_operation_set = self._hw_config.get_operations_with_unified_scales()

        self._additional_unified_scale_op_scopes = additional_unified_scale_op_scopes

        # Will handle the "wildcard" quantization situation for the time being
        if default_qconfig_list is not None:
            for op_meta, qconf_list in self._operator_allowed_qconfigs_map.items():
                trait = self._operator_quantization_trait_map.get(op_meta, QuantizationTrait.NON_QUANTIZABLE)
                if (
                    trait == QuantizationTrait.INPUTS_QUANTIZABLE
                    and HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconf_list)
                ):
                    self._operator_allowed_qconfigs_map[op_meta] = default_qconfig_list
        self._active_propagating_quantizers_queue = deque()
        self._finished_propagating_quantizers: List[PropagatingQuantizer] = []
        self._quantizers_waiting_for_branch_merge = QuantizersWaitingForMergeManager()

        self._potential_quantizers = {}
        self._num_potential_quantized_activations = 0
        self._quantizable_layer_nodes = quantizable_layer_nodes
        self._post_processing_marker_metatypes = post_processing_marker_metatypes
        self._metatypes_to_ignore = metatypes_to_ignore
        self._scales_unification_map = scales_unification_map

    def _filter_by_weight_ignored_target_scopes(
        self,
        quantizable_layer_nodes: List[QuantizableWeightedLayerNode],
        weight_ignored_scopes: List[str],
        weight_target_scopes: List[str],
    ) -> Dict[NNCFNodeName, List[QuantizerConfig]]:
        if quantizable_layer_nodes is None:
            return {}

        weight_quantizable_node_names_vs_qconfigs = {}
        for x in quantizable_layer_nodes:
            node_name = x.node.node_name
            if should_consider_scope(
                node_name, ignored_scopes=weight_ignored_scopes, target_scopes=weight_target_scopes
            ):
                weight_quantizable_node_names_vs_qconfigs[node_name] = x.qconfig_list
            else:
                nncf_logger.debug(f"Ignored adding weight quantizer for: {node_name}")
        return weight_quantizable_node_names_vs_qconfigs

    def run_on_ip_graph(
        self, ip_graph: InsertionPointGraph, metatypes_for_filter: Optional[List[OperatorMetatype]] = None
    ) -> QuantizationProposal:
        """
        The main function to be used on an InsertionPointGraph to produce
        the list of insertion commands and configs corresponding to the desired quantized
        graph state. The result of the function is not final, as it will define multiple
        possible quantizer configuration for each weight and activation quantization locations;
        a single configuration for each location must be chosen using external means.

        :param ip_graph: The InsertionPointGraph, potentially with fused operations w.r.t. the
        original model graph. The propagating quantizers will travel along the pre- and post-
        hook nodes registered in this graph.
        :param metatypes_for_filter: Metatypes are used for the removal criterion.
        :return: The intermediate propagation state in the form of QuantizationProposal, which
        defines unambiguously the locations of the propagating quantizers, but not the final
        configurations.
        """
        self._num_potential_quantized_activations = 0
        quant_prop_graph = QuantizerPropagationStateGraph(ip_graph, self._ignored_scopes, self._target_scopes)
        if self._metatypes_to_ignore is not None:
            for metatype in self._metatypes_to_ignore:
                for node_key in quant_prop_graph.get_node_keys_by_metatype(metatype):
                    self._add_node_to_ignored(node_key, quant_prop_graph)
        if self._post_processing_marker_metatypes is not None:
            post_processing_node_locator = PostprocessingNodeLocator(
                quant_prop_graph, self._quantizable_layer_nodes, self._post_processing_marker_metatypes
            )
            post_processing_node_keys = post_processing_node_locator.get_post_processing_node_keys()
            for post_processing_node_key in post_processing_node_keys:
                self._add_node_to_ignored(post_processing_node_key, quant_prop_graph)
        quant_prop_graph = self.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
        quant_prop_graph = self.setup_initial_quantizers(quant_prop_graph)

        if self._run_consistency_checks:
            quant_prop_graph.run_consistency_check()

        iteration_counter = 0
        while self._active_propagating_quantizers_queue:
            if self._visualizer is not None:
                self._visualizer.visualize_quantizer_propagation(self, quant_prop_graph, str(iteration_counter))
            if self._run_consistency_checks:
                quant_prop_graph.run_consistency_check()
            prop_quantizer = self._active_propagating_quantizers_queue.pop()
            quant_prop_graph = self.propagation_step(prop_quantizer, quant_prop_graph)
            iteration_counter += 1

        quant_prop_graph = self._filter_integer_input_quantizers(quant_prop_graph)
        if metatypes_for_filter:
            quant_prop_graph = self._filter_quantizers_by_metatypes(quant_prop_graph, metatypes_for_filter)

        if self._visualizer is not None:
            self._visualizer.visualize_quantizer_propagation(self, quant_prop_graph, "proposed")

        if self._run_consistency_checks:
            quant_prop_graph.run_consistency_check()

        for node_key in ip_graph:
            node = ip_graph.nodes[node_key]
            if node.get(InsertionPointGraph.IS_MERGED_NODE_ATTR, False):
                merged_nncf_nodes = node[InsertionPointGraph.MERGED_NNCF_NODE_LIST_NODE_ATTR]
                # If first op in fused pattern has weights, then they should be quantized
                for node in merged_nncf_nodes[1:]:
                    if node.node_name in self._weight_quantizable_node_names_vs_qconfigs:
                        self._weight_quantizable_node_names_vs_qconfigs.pop(node.node_name)

        quantizer_setup = quant_prop_graph.create_quantizer_setup(self._weight_quantizable_node_names_vs_qconfigs)
        insertions_vs_associated_prop_quants = self._map_quantization_points_to_prop_quantizers(
            self._finished_propagating_quantizers, quant_prop_graph, quantizer_setup
        )

        return QuantizationProposal(
            quantizer_setup=quantizer_setup,
            quant_prop_graph=quant_prop_graph,
            quantization_point_id_vs_prop_quantizer=insertions_vs_associated_prop_quants,
        )

    def _add_node_to_ignored(self, node_key: str, quant_prop_graph: QuantizerPropagationStateGraph) -> None:
        quant_prop_graph.ignored_node_keys[node_key] = IgnoreReason.AUTOGENERATED
        quant_prop_graph.nodes[node_key][quant_prop_graph.IS_IN_IGNORED_SCOPES] = True
        # If node has weights, also remove the weight quantizers
        underlying_nncf_nodes = quant_prop_graph.op_node_keys_to_underlying_nodes_mapping[node_key]
        for node in underlying_nncf_nodes:
            if node.node_name in self._weight_quantizable_node_names_vs_qconfigs:
                self._weight_quantizable_node_names_vs_qconfigs.pop(node.node_name)

    def _map_quantization_points_to_prop_quantizers(
        self,
        prop_quant_list: List[PropagatingQuantizer],
        quant_prop_graph: QuantizerPropagationStateGraph,
        quantizer_setup: MultiConfigQuantizerSetup,
    ) -> Dict[QuantizationPointId, PropagatingQuantizer]:
        qps_vs_associated_prop_quants_dict: Dict[QuantizationPointId, PropagatingQuantizer] = {}

        for finished_prop_quantizer in prop_quant_list:
            qip = quant_prop_graph.get_quant_insertion_point_for_propagating_quantizer(finished_prop_quantizer)
            for qp_id, qp in quantizer_setup.quantization_points.items():
                if qp.insertion_point == qip:
                    qps_vs_associated_prop_quants_dict[qp_id] = finished_prop_quantizer

        return qps_vs_associated_prop_quants_dict

    def get_final_quantizer_setup(
        self, finalized_quantization_proposal: FinalizedQuantizationProposal
    ) -> SingleConfigQuantizerSetup:
        """
        Merges consequent quantizers which ended up having the same quantization configuration.
        :param finalized_quantization_proposal:
        :return:
        """
        quant_prop_graph = finalized_quantization_proposal.quant_prop_graph
        quant_prop_graph.merge_redundant_subsequent_quantizers_across_graph()

        if self._visualizer is not None:
            self._visualizer.visualize_quantizer_propagation(self, quant_prop_graph, "final")

        if self._run_consistency_checks:
            quant_prop_graph.run_consistency_check()

        final_weight_quantizable_node_names_vs_qconfig_dict = {}
        for qp in finalized_quantization_proposal.single_config_quantizer_setup.quantization_points.values():
            if qp.is_weight_quantization_point():
                final_weight_quantizable_node_names_vs_qconfig_dict[qp.insertion_point.target_node_name] = [
                    qp.qconfig
                ]  # sic!

        if Counter(final_weight_quantizable_node_names_vs_qconfig_dict.keys()) != Counter(
            self._weight_quantizable_node_names_vs_qconfigs.keys()
        ):
            raise nncf.InternalError("Final weight quantizer setup is inconsistent with initial solver assumptions!")

        multi_setup_with_one_config_per_point = quant_prop_graph.create_quantizer_setup(
            final_weight_quantizable_node_names_vs_qconfig_dict
        )
        final_setup = multi_setup_with_one_config_per_point.select_first_qconfig_for_each_point()
        return final_setup

    def get_num_potential_quantized_activations(self) -> int:
        return self._num_potential_quantized_activations

    def _handle_quantizer_merge(
        self,
        waiting_pqs: Set[PropagatingQuantizer],
        quant_prop_graph: QuantizerPropagationStateGraph,
        branching_node_key: str,
    ):
        waiting_pqs_list = list(waiting_pqs)
        merged_pqs = []
        unmerged_pqs = []
        abort_merge = False
        for pq in waiting_pqs_list:
            # While the quantizers were waiting for the merge, one of the concat nodes
            # that will be affected by the merge may have been determined to be unquantizable.
            # Need another check for that.
            sts = self.check_branching_transition(quant_prop_graph, pq, branching_node_key)
            if sts is TransitionStatus.SHOULD_NOT_TRANSITION:
                abort_merge = True
        if not abort_merge:
            # All quantizers that are dominated by the current branching node are waiting
            # for the merge - should merge them now
            nncf_logger.debug(f"Merging PQs: {','.join([str(pq.id) for pq in waiting_pqs_list])}")
            qconfs_list = [pq.potential_quant_configs for pq in waiting_pqs_list]
            merged_qconf_list, branch_qconf_lists = self.get_merged_qconfigs_for_downward_branching_case(qconfs_list)

            if merged_qconf_list is None and self._propagation_strategy == QuantizerPropagationRule.MERGE_ALL_IN_ONE:
                all_confs = "\n".join(", ".join([f"[{str(qconf)}]" for qconf in qconfs]) for qconfs in qconfs_list)
                nncf_logger.debug(
                    f"Could not merge the quantizers at branching point {branching_node_key} - "
                    f"no common quantizer configurations found among the following: \n{all_confs}"
                )

            merge_pqs = quant_prop_graph.merge_quantizers_for_branching_node(
                waiting_pqs_list, merged_qconf_list, branch_qconf_lists, branching_node_key
            )
            for idx, qconf_list in enumerate(branch_qconf_lists):
                if qconf_list is None:
                    merged_pqs.append(waiting_pqs_list[idx])
                else:
                    unmerged_pqs.append(waiting_pqs_list[idx])
        else:
            nncf_logger.debug(f"Merge aborted for PQs {','.join([str(pq.id) for pq in waiting_pqs_list])}")
            merge_pqs = []
            unmerged_pqs = waiting_pqs_list

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

        if merge_pqs:
            self._active_propagating_quantizers_queue.extendleft(merge_pqs)
        self._quantizers_waiting_for_branch_merge.resolve_merged_node(branching_node_key)

    def propagation_step(
        self, curr_prop_quantizer: PropagatingQuantizer, quant_prop_graph: QuantizerPropagationStateGraph
    ) -> QuantizerPropagationStateGraph:
        """
        Returns an updated curr_prop_quantizer state if the quantizer is not
        yet in its final (accepting) position, and None if the quantizer is in its
        final location.  The location before and after the step should correspond to
        some insertion point.

        :param curr_prop_quantizer: The PropagatingQuantizer to currently be propagated.
        :param quant_prop_graph: The propagation state graph for `curr_prop_quantizer` to be propagated in.
        :return: The new state of `quant_prop_graph` with `curr_prop_quantizer` propagated one step further.
        """

        curr_node_key = curr_prop_quantizer.current_location_node_key
        curr_node = quant_prop_graph.nodes[curr_node_key]
        curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
        assert QuantizerPropagationStateGraph.is_insertion_point(curr_node_type)

        if curr_prop_quantizer in self._quantizers_waiting_for_branch_merge:
            branching_node_key = self._quantizers_waiting_for_branch_merge.get_blocking_node(curr_prop_quantizer)
            dom_pqs = quant_prop_graph.get_propagating_quantizers_immediately_dominated_by_node(branching_node_key)
            active_dom_pqs = set(
                filter(lambda x: x in self._active_propagating_quantizers_queue or x is curr_prop_quantizer, dom_pqs)
            )
            waiting_pqs = self._quantizers_waiting_for_branch_merge.get_waiting_quantizers_for_branching_node_key(
                branching_node_key
            )
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
            # May have ended up right below an embedding node without inputs. The path to the dominating IP will
            # not be generated (because a no-input node cannot be assigned pre-hooks), so need to directly check
            # the predecessors.
            wao_pred_node_keys = quant_prop_graph.get_predecessor_weight_as_outputs_node_keys(curr_node_key)
            if wao_pred_node_keys:
                assert len(wao_pred_node_keys) == 1
                quant_prop_graph.mark_act_quantizer_as_dependent_on_weights(curr_prop_quantizer, wao_pred_node_keys[0])

            prop_quantizer = quant_prop_graph.backtrack_propagation_until_accepting_location(curr_prop_quantizer)
            if prop_quantizer is not None:
                self._finished_propagating_quantizers.append(prop_quantizer)
            return quant_prop_graph

        surviving_prop_quantizers = []

        prop_quantizers_to_process = []
        did_clone = False

        # TODO (vshampor): include information on unified scale type in grouping; for now assuming that
        # only concat unified scale groups appear here
        unified_scale_grouped_paths = (
            quant_prop_graph.get_paths_to_immediately_dominating_insertion_points_grouped_by_unified_scales(
                curr_node_key, self._unified_scales_operation_set, self._scales_unification_map
            )
        )

        unified_scale_path_groups_vs_pqs = {k: [] for k in unified_scale_grouped_paths if k is not None}
        existing_pq_assigned = False
        for gid, path_group in unified_scale_grouped_paths.items():
            for _ in path_group:
                if existing_pq_assigned:
                    pq = quant_prop_graph.clone_propagating_quantizer(curr_prop_quantizer)
                    did_clone = True
                else:
                    pq = curr_prop_quantizer
                    existing_pq_assigned = True

                prop_quantizers_to_process.append(pq)
                if gid is not None:  # None stands for non-unified scale quantizers
                    unified_scale_path_groups_vs_pqs[gid].append(pq)

        for pq_group in unified_scale_path_groups_vs_pqs.values():
            primary_pq = pq_group[0]
            # TODO (vshampor): smarter type assignment
            primary_pq.unified_scale_type = UnifiedScaleType.UNIFY_ONLY_PER_TENSOR
            for pq in pq_group[1:]:
                quant_prop_graph.unify_pq_scales(primary_pq, pq)

        cloned_prop_quantizers = prop_quantizers_to_process if did_clone else None

        pqs_and_paths = zip(paths, prop_quantizers_to_process)
        for path, prop_quantizer in pqs_and_paths:
            status = self.check_transition_via_path(prop_quantizer, path, quant_prop_graph, cloned_prop_quantizers)
            if status == TransitionStatus.SHOULD_NOT_TRANSITION:
                if did_clone and prop_quantizer is not curr_prop_quantizer:
                    quant_prop_graph.remove_propagating_quantizer(
                        prop_quantizer, keep_propagating_quantizer_at_current_node=True
                    )
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
                self._quantizers_waiting_for_branch_merge.add_propagating_quantizer_to_wait_on_node_key(
                    prop_quantizer, branching_node_key
                )
                surviving_prop_quantizers.append(prop_quantizer)

        for prop_quantizer in surviving_prop_quantizers:
            self._active_propagating_quantizers_queue.appendleft(prop_quantizer)
        return quant_prop_graph

    def get_allowed_quantizer_configs_for_operator(self, quant_det_id: OperatorMetatype) -> List[QuantizerConfig]:
        """
        Returns the quantizer configurations that were determined as allowed for a
        given metatype by HW config or other means.

        :param quant_det_id: The metatype of the operation.
        :return: The list of allowed quantizer configurations.
        """
        return self._operator_allowed_qconfigs_map.get(quant_det_id, [])

    def set_allowed_quantization_types_for_operator_nodes(
        self, quant_prop_graph: QuantizerPropagationStateGraph
    ) -> QuantizerPropagationStateGraph:
        """
        Marks the operator nodes in the quantizer propagation state graph with
        correct quantization types based on the type of operation, HW config and/or
        other considerations.

        :param quant_prop_graph: The quantizer propagation state graph.
        :return: The same quantizer propagation state graph where operations are marked with a corresponding
          quantization type.
        """
        for node_key, node in quant_prop_graph.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                if node[QuantizerPropagationStateGraph.IS_IN_IGNORED_SCOPES]:
                    trait = QuantizationTrait.NON_QUANTIZABLE
                    node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] = trait
                    continue

                quant_det_id = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                if quant_det_id is None:
                    nncf_logger.debug(f"Unknown metatype for operator node: {node_key}")
                trait = self._operator_quantization_trait_map.get(quant_det_id, QuantizationTrait.NON_QUANTIZABLE)
                node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] = trait
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    node[QuantizerPropagationStateGraph.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR] = (
                        self.get_allowed_quantizer_configs_for_operator(quant_det_id)
                    )
        return quant_prop_graph

    def get_operator_quantization_traits_map(self) -> Dict[OperatorMetatype, QuantizationTrait]:
        """
        :return: A mapping of operator metatypes to the quantization traits to be assigned to such operations.
        """
        # TODO (vshampor): ensure that there are no name collisions between ops in different torch subpackages with
        #  the same name
        retval = {}
        if self._hw_config is None:
            for trait, meta_list in self._default_trait_to_metatype_map.items():
                for op_meta in meta_list:
                    retval[op_meta] = trait
        else:
            op_meta_vs_qconfs_map = self._hw_config.get_metatype_vs_quantizer_configs_map()
            for op_meta, qconf_list in op_meta_vs_qconfs_map.items():
                if HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconf_list):
                    trait = self._get_trait_for_op_meta_not_specified_in_hw_config(op_meta)
                elif HWConfig.is_wildcard_quantization(qconf_list):
                    for default_trait, meta_list in self._default_trait_to_metatype_map.items():
                        if op_meta in meta_list:
                            trait = default_trait
                            break
                    else:
                        trait = QuantizationTrait.NON_QUANTIZABLE
                else:
                    trait = QuantizationTrait.INPUTS_QUANTIZABLE
                retval[op_meta] = trait
        return retval

    def _get_trait_for_op_meta_not_specified_in_hw_config(self, op_meta: OperatorMetatype) -> QuantizationTrait:
        if not op_meta.hw_config_names:
            # The metatype might not have an associated name in the config
            # namespace (yet) - use default trait
            for default_trait, meta_list in self._default_trait_to_metatype_map.items():
                if op_meta in meta_list:
                    trait = default_trait
                    break
            else:
                trait = QuantizationTrait.NON_QUANTIZABLE
                nncf_logger.debug(
                    f"Operation metatype {op_meta} encountered, but it has no default "
                    f"quantization trait and the HW config entry is not given for it - "
                    f"assuming non-quantizable."
                )
        else:
            # There IS a valid HW config name for the metatype, but it is deliberately not specified
            # in the config, which means that it should execute in FP32
            trait = QuantizationTrait.NON_QUANTIZABLE

        return trait

    def _get_operator_qconfigs_map(self) -> Dict[OperatorMetatype, List[QuantizerConfig]]:
        # TODO (vshampor): ensure that there are no name collisions between ops in different torch subpackages
        #  with the same name
        retval = {}  # Metas not in retval will correspond to wildcard quantization
        if self._hw_config is None:
            for trait, meta_list in self._default_trait_to_metatype_map.items():
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    for op_meta in meta_list:
                        if self.default_global_qconfig_list is not None:
                            retval[op_meta] = deepcopy(self.default_global_qconfig_list)
                        else:
                            retval[op_meta] = deepcopy(self.DEFAULT_QUANTIZATION_TYPES)
                elif trait == QuantizationTrait.NON_QUANTIZABLE:
                    for op_meta in meta_list:
                        retval[op_meta] = None
        else:
            retval = self._hw_config.get_metatype_vs_quantizer_configs_map()
        return retval

    def debug_visualize(self, quant_prop_graph: QuantizerPropagationStateGraph, dump_path: str):
        """
        Visualizes in a .dot format the state of the current quantizer propagation state graph and
        the associated solver information.

        :param quant_prop_graph: The propagation state graph to visualize.
        :param dump_path: The name of the output .dot file.
        """
        out_graph = quant_prop_graph.get_visualized_graph()
        active_ids_str = ", ".join([str(pq.id) for pq in self._active_propagating_quantizers_queue])
        finished_ids_str = ", ".join([str(pq.id) for pq in self._finished_propagating_quantizers])
        next_id_str = ""
        if self._active_propagating_quantizers_queue:
            next_id_str = str(self._active_propagating_quantizers_queue[-1].id)
        out_graph.graph["graph"] = {
            "label": "Propagating quantizers: {}\n"
            "Next quantizer to be propagated: {}\n"
            "Finished quantizers: {}".format(active_ids_str, next_id_str, finished_ids_str),
            "labelloc": "t",
        }
        pth = deepcopy(dump_path)
        write_dot_graph(out_graph, pth)

    def setup_initial_quantizers(
        self, quant_prop_graph: QuantizerPropagationStateGraph
    ) -> QuantizerPropagationStateGraph:
        """
        Determines the initial subset of the nodes that must be quantized
        and corresponding allowed quantization configs (possibly multiple) for each
        quantizable operation, and sets up propagating quantizers at initial locations.

        :param quant_prop_graph: The quantizer propagation state graph without any
          quantizers registered.
        :return: The same state graph with each operation requiring quantized inputs
          having a quantizer registered for each of its inputs and placed into a pre-hook
          operation spot corresponding to the input.
        """
        for node_key in nx.lexicographical_topological_sort(quant_prop_graph):
            node = quant_prop_graph.nodes[node_key]
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                num_input_activations = quant_prop_graph.get_num_input_activations(node_key)
                self._num_potential_quantized_activations += num_input_activations
                if node_key in quant_prop_graph.ignored_node_keys:
                    msg = f"Not adding activation input quantizer for operation: {node_key}"
                    if quant_prop_graph.ignored_node_keys[node_key] == IgnoreReason.AUTOGENERATED:
                        nncf_logger.debug(msg)
                    else:
                        nncf_logger.info(msg)
                    continue
                self._setup_initial_quantizers_for_operator_node(node_key, quant_prop_graph)

        if self._additional_unified_scale_op_scopes is not None:
            # Link the prop quantizers according to specification in NNCF config
            occupied_insertion_points_vs_pqs: Dict[TargetPoint, PropagatingQuantizer] = {}
            for pq in self._active_propagating_quantizers_queue:
                ip_node_key = pq.current_location_node_key
                ip_node = quant_prop_graph.nodes[ip_node_key]
                assert QuantizerPropagationStateGraph.is_insertion_point(
                    ip_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                )
                ip = ip_node[QuantizerPropagationStateGraph.QUANT_INSERTION_POINT_DATA_NODE_ATTR]
                occupied_insertion_points_vs_pqs[ip] = pq
            coalesced_ips = self.coalesce_insertion_points(
                list(occupied_insertion_points_vs_pqs.keys()), self._additional_unified_scale_op_scopes
            )
            for linked_ip_group in coalesced_ips:
                if len(linked_ip_group) <= 2:
                    continue
                main_ip = linked_ip_group[0]
                main_pq = occupied_insertion_points_vs_pqs[main_ip]

                for ip in linked_ip_group[1:]:
                    pq = occupied_insertion_points_vs_pqs[ip]
                    quant_prop_graph.unify_pq_scales(main_pq, pq)

        return quant_prop_graph

    @staticmethod
    def coalesce_insertion_points(
        target_insertion_points: List[TargetPoint], linked_scopes_groups_list: List[List[str]]
    ) -> List[List[TargetPoint]]:
        """
        Accepts a list of TargetPoints and groups these according to linked_scope_groups_list.
        The matching of a TargetPoint to the string entries in the lists is made based on the
        string representation of TargetPoint's target_node_name attribute.

        :param target_insertion_points: A list of TargetPoint objects to be grouped based on the
          `linked_scopes_groups_list`
        :param linked_scopes_groups_list: A list of string lists, where each list defines a desired grouping
          of TargetPoints, and each string of the list is matched against the string representation of
          corresponding TargetPoints.
        :return: A list of TargetPoint groups; each group is a list of TargetPoint's.
        """

        if linked_scopes_groups_list is None:
            return [
                [
                    ip,
                ]
                for ip in target_insertion_points
            ]
        retval = []
        insertion_point_indices_vs_group_id = OrderedDict()

        for group_idx, group_list in enumerate(linked_scopes_groups_list):
            for group_member_node_name in group_list:
                matching_indices = list(
                    filter(
                        lambda x: target_insertion_points[x].target_node_name == group_member_node_name,
                        range(len(target_insertion_points)),
                    )
                )
                if len(matching_indices) == 0:
                    raise nncf.ValidationError(
                        "No match for linked quantizer entry {} among activation quantizers!".format(
                            group_member_node_name
                        )
                    )

                for target_idx in matching_indices:
                    if target_idx in insertion_point_indices_vs_group_id:
                        raise nncf.InternalError(
                            "Linked activation quantizer groups {} and {} "
                            "overlap!".format(group_idx, insertion_point_indices_vs_group_id[target_idx])
                        )
                for target_idx in matching_indices:
                    insertion_point_indices_vs_group_id[target_idx] = group_idx

        for i in range(len(target_insertion_points)):
            if i not in insertion_point_indices_vs_group_id:
                insertion_point_indices_vs_group_id[i] = None

        group_indices_list: List[List[int]] = [[] for _ in linked_scopes_groups_list]
        for insertion_point_idx, group_idx in insertion_point_indices_vs_group_id.items():
            if group_idx is not None:
                group_indices_list[group_idx].append(insertion_point_idx)

        for intra_group_indices in group_indices_list:
            main_ip_idx = intra_group_indices[0]
            main_ip = target_insertion_points[main_ip_idx]
            grouped_list = [
                main_ip,
            ]
            for linked_ip_idx in intra_group_indices[1:]:
                grouped_list.append(target_insertion_points[linked_ip_idx])
            retval.append(grouped_list)

        for insertion_point_idx, group_idx in insertion_point_indices_vs_group_id.items():
            if group_idx is None:
                retval.append(
                    [
                        target_insertion_points[insertion_point_idx],
                    ]
                )

        return retval

    def _filter_qconfigs_according_to_scope(
        self, qconf_list: List[QuantizerConfig], nncf_node_name: NNCFNodeName
    ) -> List[QuantizerConfig]:
        if self._global_constraints is not None:
            local_constraints = self._global_constraints[QuantizerGroup.ACTIVATIONS]
        else:
            local_constraints = QuantizationConstraints()

        act_scope_overrides = self._scope_overrides.get("activations", {})
        for overridden_scope, scoped_override_dict in act_scope_overrides.items():
            if matches_any(nncf_node_name, overridden_scope):
                scope_constraints = QuantizationConstraints.from_config_dict(scoped_override_dict)
                local_constraints = local_constraints.get_updated_constraints(scope_constraints)

        if self._hw_config is not None:
            constrained_config_list = local_constraints.constrain_qconfig_list(
                nncf_node_name, self._hw_config.target_device, qconf_list
            )
        else:
            constrained_config_list = [local_constraints.apply_constraints_to(qconfig) for qconfig in qconf_list]

        return constrained_config_list

    def _setup_initial_quantizers_for_operator_node(
        self, operator_node_key: str, quant_prop_graph: QuantizerPropagationStateGraph
    ):
        node = quant_prop_graph.nodes[operator_node_key]

        # preds are in sorted order for reproducibility
        preds = list(sorted(quant_prop_graph.predecessors(operator_node_key)))

        if not preds:
            return  # TODO (vshampor): remove this once module insertion points are included in the IP graph

        metatype = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
        if not self._quantize_outputs and metatype in OUTPUT_NOOP_METATYPES:
            return
        # No need to place quantizers for FP32-forced ops, naturally
        if (
            node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
            in [QuantizationTrait.NON_QUANTIZABLE, QuantizationTrait.QUANTIZATION_AGNOSTIC, QuantizationTrait.CONCAT]
            and metatype not in OUTPUT_NOOP_METATYPES
        ):
            return
        qconf_list = self.get_allowed_quantizer_configs_for_operator(metatype)
        if metatype in OUTPUT_NOOP_METATYPES:
            qconf_list = deepcopy(self.default_global_qconfig_list)
        assert qconf_list is not None

        if not HWConfig.is_wildcard_quantization(qconf_list):
            nncf_node_ref = next(iter(quant_prop_graph.op_node_keys_to_underlying_nodes_mapping[operator_node_key]))
            qconf_list = self._filter_qconfigs_according_to_scope(qconf_list, nncf_node_ref.node_name)
        else:
            qconf_list = [deepcopy(DEFAULT_QUANTIZER_CONFIG)]

        is_unified_scale = metatype in self._unified_scales_operation_set
        if is_unified_scale:
            # Filtering out the per-channel cases in the unified scale scenario.
            # In order to support unified per-channel scales, we will need to handle a situation
            # when a per-channel shared (unified) quantizer on one branch passes a shape-changing
            # operation such as `view` or `transpose`, and the linked unified quantizer does not do
            # so due to one or the other reason; the per-channel scale shapes to be applied therefore
            # will be different.
            # TODO (vshampor): What possibly needs to be done in this direction:
            # 1. keep ForwardTraceOnly ops in graph after all, to be able to track shape changes
            # 2. transpose input tensors to the quantization modules on the fly to accommodate scale,
            #    or vice versa, transpose scale to accommodate shape; need to handle exporting as well
            per_tensor_qconf_list = list(filter(lambda x: x.per_channel is False, qconf_list))
            op_meta_name = metatype.__class__.__name__
            if len(per_tensor_qconf_list) != len(qconf_list):
                if not per_tensor_qconf_list:
                    raise nncf.InternalError(
                        "Unified scales currently do not support per-channel configuration - dropping"
                        "per-channel configuration options for {} resulted in no valid quantization "
                        "configs!".format(op_meta_name)
                    )
                nncf_logger.warning(
                    f"Unified scales currently do not support per-channel configuration - dropping"
                    f"per-channel configuration options for {op_meta_name}"
                )
                qconf_list = per_tensor_qconf_list

        pred_ip_key_vs_qconf_dict = OrderedDict()
        # Should be immediately preceded by insertion points (pre-hook)
        for pred_ip_key in preds:
            pred_node = quant_prop_graph.nodes[pred_ip_key]
            pred_node_type = pred_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            assert QuantizerPropagationStateGraph.is_insertion_point(
                pred_node_type
            ), "Invalid insertion point graph supplied for quantizer propagation!"

            ip = pred_node[QuantizerPropagationStateGraph.QUANT_INSERTION_POINT_DATA_NODE_ATTR]
            input_port_id = ip.input_port_id
            if input_port_id in metatype.ignored_input_ports:
                continue

            if metatype.target_input_ports is not None and input_port_id not in metatype.target_input_ports:
                continue

            edge = quant_prop_graph.edges[pred_ip_key, operator_node_key]
            if not edge[QuantizerPropagationStateGraph.IS_INTEGER_PATH_EDGE_ATTR]:
                pred_ip_key_vs_qconf_dict[pred_ip_key] = qconf_list
            else:
                nncf_logger.debug(f"Detected integer input {pred_ip_key} - won't set up a propagating quantizer for it")

        if not pred_ip_key_vs_qconf_dict:
            # All inputs to the operator were integer
            return

        # Cloning a single propagating quantizer onto all node inputs - revise if separate
        # quantizer configuration for different inputs is required
        pred_ip_key_vs_qconf_list = list(iter(pred_ip_key_vs_qconf_dict.items()))
        main_pq_ip_key, main_pq_qconf_list = pred_ip_key_vs_qconf_list[0]
        main_prop_quantizer = quant_prop_graph.add_propagating_quantizer(
            main_pq_qconf_list,
            main_pq_ip_key,
            unified_scale_type=UnifiedScaleType.UNIFY_ALWAYS if is_unified_scale else None,
        )
        main_prop_quantizer.last_accepting_location_node_key = main_pq_ip_key
        self._active_propagating_quantizers_queue.appendleft(main_prop_quantizer)

        main_pq_gid = None

        if is_unified_scale:
            main_pq_gid = quant_prop_graph.get_unified_scale_group_id_by_propagating_quantizer_id(
                main_prop_quantizer.id
            )

        for additional_pq_ip_key, _ in pred_ip_key_vs_qconf_list[1:]:
            additional_pq = quant_prop_graph.add_propagating_quantizer(
                main_pq_qconf_list,
                additional_pq_ip_key,
                unified_scale_type=UnifiedScaleType.UNIFY_ALWAYS if is_unified_scale else None,
                unified_scale_group_id_override=main_pq_gid,
            )
            additional_pq.last_accepting_location_node_key = additional_pq_ip_key
            self._active_propagating_quantizers_queue.appendleft(additional_pq)

    def check_branching_transition(
        self,
        quant_prop_graph: QuantizerPropagationStateGraph,
        prop_quant_to_transition: PropagatingQuantizer,
        branching_node_key: str,
    ) -> TransitionStatus:
        """
        If a propagating quantizer advances through a node that branches
        downwards, the branches neighbouring to the one that the propagating quantizer
        had just propagated from will have the precision of the quantizer imposed upon
        them.  This is not always desirable - we might want to keep some branches in
        higher precision than the others. For this reason, this function checks whether
        the quantizer may safely advance through a branching node based on the possible
        configs of the quantizers it might affect by doing so.

        :param quant_prop_graph: The current quantizer propagation state graph.
        :param prop_quant_to_transition: The propagating quantizer that is about to transition
          upwards through a branching node.
        :param branching_node_key: The node key in `quant_prop_graph` corresponding to the node
          that branches downwards.
        :return: The TransitionStatus indicating in which fashion the transition should occur.
        """
        is_dominating_outputs = quant_prop_graph.is_branching_node_dominating_outputs(branching_node_key)
        if is_dominating_outputs and not self._quantize_outputs:
            return TransitionStatus.SHOULD_NOT_TRANSITION

        dom_op_node_keys = quant_prop_graph.get_non_quant_agnostic_op_nodes_immediately_dominated_by_node(
            branching_node_key
        )
        dom_op_quantizers = set()
        for op_node_key in dom_op_node_keys:
            op_node = quant_prop_graph.nodes[op_node_key]

            # Check all branches have a quantizer on it before the merge
            if op_node["op_meta"].target_input_ports is not None:
                all_branches_are_quantized = quant_prop_graph.all_outputs_are_quantized(branching_node_key)
                if not all_branches_are_quantized:
                    return TransitionStatus.SHOULD_NOT_TRANSITION

            trait = op_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
            affecting_prop_quantizers = op_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if affecting_prop_quantizers:
                for aff_pq in affecting_prop_quantizers:
                    dom_op_quantizers.add(aff_pq)
            else:
                if trait is not QuantizationTrait.CONCAT:
                    # The branch op is forced to be FP32 - should not proceed through the branch node.
                    return TransitionStatus.SHOULD_NOT_TRANSITION

                # Have to determine if the concat node will potentially have input quantization applied
                # as a result of further propagation.
                pqs_dominated_by_cat = quant_prop_graph.get_propagating_quantizers_immediately_dominated_by_node(
                    op_node_key
                )
                active_pqs_dominated_by_cat = set(
                    filter(lambda x: x in self._active_propagating_quantizers_queue, pqs_dominated_by_cat)
                )
                if not active_pqs_dominated_by_cat:
                    # There is no chance for this concat node to be quantized later,
                    # should not attempt merge.
                    return TransitionStatus.SHOULD_NOT_TRANSITION
                # There are still some quantizers that may propagate upwards through this concat node
                # and ultimately lead to the concat node having quantized inputs
                dom_op_quantizers.update(active_pqs_dominated_by_cat)

        dom_op_quantizers.discard(prop_quant_to_transition)
        if dom_op_quantizers:
            return TransitionStatus.SHOULD_WAIT_FOR_MERGE

        return TransitionStatus.SHOULD_TRANSITION

    def _check_affecting_quantizers_in_common_path(
        self, affecting_quantizers: List[PropagatingQuantizer], cloned_prop_quantizers: List[PropagatingQuantizer]
    ):
        # Handling the case where multiple freshly cloned quantizers have to follow paths that are different,
        # but have a common edge or node
        safe_affecting_quantizers = [pq for pq in affecting_quantizers if pq in cloned_prop_quantizers]
        assert safe_affecting_quantizers == affecting_quantizers

    def _check_for_affecting_quantizer_conflicts(
        self,
        curr_prop_quantizer: PropagatingQuantizer,
        affecting_quantizers: List[PropagatingQuantizer],
        cloned_prop_quantizers: Optional[List[PropagatingQuantizer]],
    ) -> Optional[TransitionStatus]:
        if cloned_prop_quantizers is not None:
            self._check_affecting_quantizers_in_common_path(affecting_quantizers, cloned_prop_quantizers)
            return None

        # Affecting quantizers should have the same configs by construction, so we only
        # check the first
        curr_pq_configs = curr_prop_quantizer.potential_quant_configs
        target_pq_configs = affecting_quantizers[0].potential_quant_configs
        if (
            curr_pq_configs == target_pq_configs
            or HWConfig.is_wildcard_quantization(curr_pq_configs)
            or HWConfig.is_wildcard_quantization(target_pq_configs)
        ):
            return TransitionStatus.SHOULD_MERGE
        return TransitionStatus.SHOULD_NOT_TRANSITION

    def check_transition_via_path(
        self,
        prop_quantizer: PropagatingQuantizer,
        path: PropagationPath,
        quant_prop_graph: QuantizerPropagationStateGraph,
        cloned_prop_quantizers: Optional[List[PropagatingQuantizer]] = None,
    ) -> TransitionStatus:
        """
        Determines which action should be taken regarding the
        prop_quantizer's propagation via path, which may be one of many possible
        propagation paths.

        :param prop_quantizer: The propagating quantizer to be currently considered.
        :param path: The path, defined in terms of `quant_prop_graph` edges, along which the
         `prop_quantizer` is supposed to propagate now.
        :param quant_prop_graph: The quantizer propagation state graph holding the `prop_quantizer`.
        :param cloned_prop_quantizers: Optional - if specified, would mean that the quantizer had to be
          cloned before transition, which impacts the logic of the function.
        :return: The status of the transition determining how it should proceed.
        """

        for from_node_key, to_node_key in path:
            from_node = quant_prop_graph.nodes[from_node_key]

            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = from_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait is QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS:
                    quant_prop_graph.mark_act_quantizer_as_dependent_on_weights(prop_quantizer, from_node_key)
                    return TransitionStatus.SHOULD_NOT_TRANSITION
                if trait in [QuantizationTrait.NON_QUANTIZABLE, QuantizationTrait.INPUTS_QUANTIZABLE]:
                    return TransitionStatus.SHOULD_NOT_TRANSITION

            edge = quant_prop_graph.edges[from_node_key, to_node_key]
            # Check if current edge to traverse corresponds to integer-valued tensors such as indices
            if edge[QuantizerPropagationStateGraph.IS_INTEGER_PATH_EDGE_ATTR]:
                return TransitionStatus.SHOULD_NOT_TRANSITION

            # Check if current edge to traverse is affected by any of the quantizers
            potential_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if potential_quantizers:
                sts = self._check_for_affecting_quantizer_conflicts(
                    prop_quantizer, potential_quantizers, cloned_prop_quantizers
                )

                if sts is not None:
                    return sts

            # Check if the target node is affected by any of the quantizers
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if QuantizerPropagationStateGraph.is_insertion_point(from_node_type):
                potential_quantizers = from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if potential_quantizers:
                    sts = self._check_for_affecting_quantizer_conflicts(
                        prop_quantizer, potential_quantizers, cloned_prop_quantizers
                    )
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
                status = self.check_branching_transition(quant_prop_graph, prop_quantizer, from_node_key)
                if status is TransitionStatus.SHOULD_NOT_TRANSITION or status is TransitionStatus.SHOULD_WAIT_FOR_MERGE:
                    return status

        return TransitionStatus.SHOULD_TRANSITION

    def get_merged_qconfigs_for_downward_branching_case(
        self, potential_qconfigs_for_each_branch: List[List[Optional[QuantizerConfig]]]
    ) -> Tuple[Optional[List[QuantizerConfig]], List[Optional[List[QuantizerConfig]]]]:
        """
        Returns a tuple, of which the first node is the qconfig list for the quantizer to be placed
        above the branching node (i.e. that will affect all of the downward branches), and a list
        of nodes which are either None (which means that the corresponding branch quantizer has been successfully
        merged, or qconfigs list to be set for the corresponding branch quantizer if it cannot be merged (e.g. if
        requantization to a lower bitwidth has to be done for this branch)

        :param potential_qconfigs_for_each_branch: For each branch defines the list of available configurations
          of the quantizer currently impacting this branch.
        :return: A tuple, out of which the first element corresponds to the allowed quantizer configurations
          of the merged quantizer, if any, and the second element corresponds to configurations of the quantizers
          that would have to remain on the branches (if any).
        """

        if self._propagation_strategy == QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES:
            # Do not merge at all
            return None, potential_qconfigs_for_each_branch
        if self._propagation_strategy == QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME:
            # Only merge for exact matches of the qconfig lists
            first_pq_list = potential_qconfigs_for_each_branch[0]
            first_pq_list_counter = Counter(first_pq_list)

            for other_pq_list in potential_qconfigs_for_each_branch[1:]:
                if first_pq_list_counter != Counter(other_pq_list):
                    return None, potential_qconfigs_for_each_branch

            return first_pq_list, [None for _ in potential_qconfigs_for_each_branch]

        # Attempt to produce a merged config options space
        qconfigs_union = set()
        for branch_qconfig_list in potential_qconfigs_for_each_branch:
            qconfigs_union.update(set(branch_qconfig_list))
        merged_qconfig_list = []

        nncf_logger.debug(f"Union of configs: {';'.join([str(qc) for qc in qconfigs_union])}")

        def compatible_with_requant(qconf: QuantizerConfig, other_qconf_list: List[QuantizerConfig]) -> bool:
            if qconf in other_qconf_list:
                return True
            for other_qconf in other_qconf_list:
                if not other_qconf.is_valid_requantization_for(qconf):
                    return False
            return True

        def compatible_wo_requant(qconf: QuantizerConfig, other_qconf_list: List[QuantizerConfig]) -> bool:
            if qconf in other_qconf_list:
                return True
            return False

        if self._propagation_strategy == QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION:
            compatible_fn = compatible_with_requant
        elif self._propagation_strategy == QuantizerPropagationRule.MERGE_ALL_IN_ONE:
            compatible_fn = compatible_wo_requant
        else:
            raise nncf.ValidationError(f"Unknown propagation strategy: {self._propagation_strategy}")

        for qconf in qconfigs_union:
            if all(compatible_fn(qconf, qconf_list) for qconf_list in potential_qconfigs_for_each_branch):
                merged_qconfig_list.append(qconf)

        nncf_logger.debug(f"Merged list before sorting: {';'.join([str(qc) for qc in merged_qconfig_list])}")

        if not merged_qconfig_list:
            # Impossible to produce a merged configuration space of any kind, won't merge
            return None, potential_qconfigs_for_each_branch

        # Sort the merged list according to an ad-hoc-calculated priority
        qconfig_and_priority_list = self.__assign_priorities_to_configs_in_merged_list(
            merged_qconfig_list, potential_qconfigs_for_each_branch
        )

        qconfig_and_priority_list_sorted_by_priority = sorted(qconfig_and_priority_list, key=lambda x: x[1])
        config_list_to_print = ";".join(
            [str(qc_tup[1]) + ":" + str(qc_tup[0]) for qc_tup in qconfig_and_priority_list_sorted_by_priority]
        )
        nncf_logger.debug(f"Priority-sorted merge qconfigs: {config_list_to_print}")

        merged_qconfig_list = self.__disambiguate_config_list(qconfig_and_priority_list_sorted_by_priority)
        nncf_logger.debug(f"Disambiguated merge qconfig list: {';'.join([str(qc) for qc in merged_qconfig_list])}")

        merged_qconfig_list_counter = Counter(merged_qconfig_list)
        resulting_branch_qconfig_lists = [None for _ in potential_qconfigs_for_each_branch]

        if self._propagation_strategy == QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION:
            for idx, branch_qconfig_list in enumerate(potential_qconfigs_for_each_branch):
                if Counter(branch_qconfig_list) == merged_qconfig_list_counter:
                    continue  # This branch will have the branch quantizer removed
                resulting_branch_qconfig_lists[idx] = branch_qconfig_list

        return merged_qconfig_list, resulting_branch_qconfig_lists

    def __assign_priorities_to_configs_in_merged_list(
        self,
        merged_qconfig_list: List[QuantizerConfig],
        potential_qconfigs_for_each_branch: List[List[QuantizerConfig]],
    ) -> List[Tuple[QuantizerConfig, int]]:
        # Basically, the original branches vote on a priority of a config in the merged
        # qconfig list based on the position of said qconfig in their own qconfig list.
        # This still does not properly disambiguate configs in all situations. Downstream code
        # takes 0-th config in the list as the final config file. Without an external, unambiguous
        # priority mechanism or manual config selection there is no way to do a consistent, branch order-independent
        # merge.
        qconfig_and_priority_list: List[Tuple[QuantizerConfig, int]] = []
        for merged_qconfig in merged_qconfig_list:
            priority = 0
            max_original_list_len = max(len(x) for x in potential_qconfigs_for_each_branch)
            for original_branch_qconfig_list in potential_qconfigs_for_each_branch:
                try:
                    idx = original_branch_qconfig_list.index(merged_qconfig)
                except ValueError:
                    # Move the configs that inevitably lead to requantization closer to the end of the list
                    idx = max_original_list_len + 1
                priority += idx
            qconfig_and_priority_list.append((merged_qconfig, priority))
        return qconfig_and_priority_list

    def __disambiguate_config_list(
        self, qconfig_list_with_priority: List[Tuple[QuantizerConfig, int]]
    ) -> List[QuantizerConfig]:
        """
        The input list should be sorted in descending order of priority. In case some qconfigs in the list have the
        same priority, this function will resolve the ambiguity in ordering these qconfigs in the final returned
        list.
        """

        class QConfigComparator:
            def __init__(self, qconfig: QuantizerConfig):
                self.qconfig = qconfig

            def __lt__(self, other: "QConfigComparator"):
                # Prefer higher bitwidths, per-tensor, symmetrical
                if self.qconfig.num_bits > other.qconfig.num_bits:
                    return True
                if self.qconfig.num_bits < other.qconfig.num_bits:
                    return False
                if self.qconfig.per_channel is False and other.qconfig.per_channel is True:
                    return True
                if self.qconfig.per_channel is True and other.qconfig.per_channel is False:
                    return False
                if (
                    self.qconfig.mode is QuantizationMode.SYMMETRIC
                    and other.qconfig.mode is QuantizationMode.ASYMMETRIC
                ):
                    return True
                if (
                    self.qconfig.mode is QuantizationMode.ASYMMETRIC
                    and other.qconfig.mode is QuantizationMode.SYMMETRIC
                ):
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

    def get_finished_propagating_quantizers(self) -> List[PropagatingQuantizer]:
        """
        :return: A list of propagating quantizers that have finished propagating.
        """
        return self._finished_propagating_quantizers

    def get_active_propagating_quantizers_queue(self) -> Deque[PropagatingQuantizer]:
        """
        :return: The queue of propagating quantizers that are still propagating.
        """
        return self._active_propagating_quantizers_queue

    def get_total_quantizer_count(self):
        return len(self.get_finished_propagating_quantizers()) + len(self.get_active_propagating_quantizers_queue())

    def _filter_integer_input_quantizers(
        self, quant_prop_graph: QuantizerPropagationStateGraph
    ) -> QuantizerPropagationStateGraph:
        input_node_vs_qid_dict = quant_prop_graph.get_quantizers_at_input_nncf_nodes()
        integer_input_quantizer_ids = set()

        for input_node, input_quantizer_ids in input_node_vs_qid_dict.items():
            assert input_node.metatype in INPUT_NOOP_METATYPES
            if input_node.is_integer_input():
                integer_input_quantizer_ids.update(set(input_quantizer_ids))

        filtered_finished_pqs = list(
            filter(lambda pq: pq.id not in integer_input_quantizer_ids, self._finished_propagating_quantizers)
        )
        integer_input_pqs = list(
            filter(lambda pq: pq.id in integer_input_quantizer_ids, self._finished_propagating_quantizers)
        )
        self._finished_propagating_quantizers = filtered_finished_pqs
        for integer_input_pq in integer_input_pqs:
            quant_prop_graph.remove_propagating_quantizer(integer_input_pq)

        return quant_prop_graph

    def _filter_quantizers_by_metatypes(
        self, quant_prop_graph: QuantizerPropagationStateGraph, metatypes: List[OperatorMetatype]
    ) -> QuantizerPropagationStateGraph:
        """
        Removes quantizers for which _is_quantizer_to_remove returns True.

        :param quant_prop_graph: The quantizer propagation state graph.
        :param metatypes: Metatypes are used for the removal criterion.
        :return: Filtered quantizer propagation state graph.
        """

        def _is_quantizer_to_remove(
            quant_prop_graph: QuantizerPropagationStateGraph,
            quantizer: PropagatingQuantizer,
            metatypes: List[OperatorMetatype],
        ) -> bool:
            """
            Returns True if the quantizer meets the criteria for removal. The criteria are as follows:
            1. The quantizer is generated from a node whose metatype is in the provided metatypes.
            2. The quantizer is not propagated.
            3. The quantizer has only one child.
            4. The quantized node generates only one activation quantizer.
            The function relies on the fact that considered metatypes should have two inputs.
            In that case, if considered node at InsertionPointGraph has only one input,
            it means that the another one is a constant.

            :param quant_prop_graph: The quantizer propagation state graph holding the `quantizer`.
            :param quantizer: The propagating quantizer to be currently considered.
            :param metatypes: Metatypes are used for the criterion.
            :return: True if quantizer satisfies the criteria, otherwise - False.
            """
            quantizer_children = quantizer.quantized_input_sink_operator_nodes
            quantized_node_metatype = quant_prop_graph.nodes[quantized_node_key][
                QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR
            ]
            quantizers_generated_for_node = quant_prop_graph.nodes[quantized_node_key][
                quant_prop_graph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR
            ]

            is_one_quantizer_generated_for_node = len(quantizers_generated_for_node) == 1
            is_one_child = len(quantizer_children) == 1
            is_metatype_to_filter = quantized_node_metatype in metatypes
            is_quantizer_not_propagated = len(quantizer.propagation_path) <= 1

            return (
                is_one_child
                and is_metatype_to_filter
                and is_one_quantizer_generated_for_node
                and is_quantizer_not_propagated
            )

        quantizers = self._finished_propagating_quantizers
        to_remove_quantizers = []
        for quantizer in quantizers:
            quantized_node_key = next(iter(quantizer.quantized_input_sink_operator_nodes))
            if _is_quantizer_to_remove(quant_prop_graph, quantizer, metatypes):
                nncf_logger.debug(f"Quantizer generated for a node {quantized_node_key} will be removed.")
                to_remove_quantizers.append(quantizer)
        for quantizer in to_remove_quantizers:
            quant_prop_graph.remove_propagating_quantizer(quantizer)
            self._finished_propagating_quantizers.remove(quantizer)
        return quant_prop_graph
