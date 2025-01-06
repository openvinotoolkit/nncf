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

from math import floor
from math import isclose
from typing import List, Set

import tensorflow as tf

import nncf
from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.common.graph import NNCFGraph
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.logging import nncf_logger
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.common.pruning.schedulers import PruningScheduler
from nncf.common.pruning.shape_pruning_processor import ShapePruningProcessor
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.pruning.statistics import PrunedModelStatistics
from nncf.common.pruning.statistics import PrunedModelTheoreticalBorderline
from nncf.common.pruning.utils import get_prunable_layers_in_out_channels
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.common.pruning.weights_flops_calculator import WeightsFlopsCalculator
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.common.utils.debug import is_debug
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.config.schemata.defaults import PRUNING_ALL_WEIGHTS
from nncf.config.schemata.defaults import PRUNING_FILTER_IMPORTANCE
from nncf.config.schemata.defaults import PRUNING_SCHEDULE
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoBuilder
from nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoController
from nncf.tensorflow.pruning.base_algorithm import PrunedLayerInfo
from nncf.tensorflow.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from nncf.tensorflow.pruning.filter_pruning.functions import calculate_binary_mask
from nncf.tensorflow.pruning.filter_pruning.functions import tensor_l2_normalizer
from nncf.tensorflow.pruning.operations import TF_PRUNING_OPERATOR_METATYPES
from nncf.tensorflow.pruning.operations import TFConvolutionPruningOp
from nncf.tensorflow.pruning.operations import TFElementwisePruningOp
from nncf.tensorflow.pruning.operations import TFLinearPruningOp
from nncf.tensorflow.pruning.operations import TFTransposeConvolutionPruningOp
from nncf.tensorflow.pruning.tensor_processor import TFNNCFPruningTensorProcessor
from nncf.tensorflow.pruning.utils import broadcast_filter_mask
from nncf.tensorflow.pruning.utils import collect_output_shapes
from nncf.tensorflow.pruning.utils import get_filter_axis
from nncf.tensorflow.pruning.utils import get_filters_num
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from nncf.tensorflow.tensor import TFNNCFTensor


@TF_COMPRESSION_ALGORITHMS.register("filter_pruning")
class FilterPruningBuilder(BasePruningAlgoBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable filter pruning during fine-tuning.
    """

    def _build_controller(self, model: tf.keras.Model):
        return FilterPruningController(
            model, self._graph, self._op_names, self._prunable_types, self._pruned_layer_groups_info, self.config
        )

    def _is_pruned_layer(self, layer: tf.keras.layers.Layer) -> bool:
        return layer.__class__.__name__ in self._prunable_types

    def _get_op_types_of_pruned_layers(self) -> List[str]:
        return [
            op_name
            for meta_op in [TFConvolutionPruningOp, TFTransposeConvolutionPruningOp, TFLinearPruningOp]
            for op_name in meta_op.get_all_op_aliases()
        ]

    def _get_types_of_grouping_ops(self) -> List[str]:
        return TFElementwisePruningOp.get_all_op_aliases()


@api()
@ADAPTIVE_COMPRESSION_CONTROLLERS.register("tf_filter_pruning")
class FilterPruningController(BasePruningAlgoController):
    """
    Controller class for the filter pruning algorithm.
    """

    def __init__(
        self,
        target_model: tf.keras.Model,
        graph: NNCFGraph,
        op_names: List[str],
        prunable_types: List[str],
        pruned_layer_groups: Clusterization[PrunedLayerInfo],
        config: NNCFConfig,
    ):
        super().__init__(target_model, op_names, prunable_types, pruned_layer_groups, config)
        self._original_graph = graph
        params = self.pruning_config.get("params", {})
        self.frozen = False
        self.pruning_quota = 0.9

        self._weights_flops_calc = WeightsFlopsCalculator(
            conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES, linear_op_metatypes=LINEAR_LAYER_METATYPES
        )

        self._shape_pruning_proc = ShapePruningProcessor(
            pruning_operations_metatype=TF_PRUNING_OPERATOR_METATYPES, prunable_types=prunable_types
        )

        self._pruning_quotas = {}
        self._next_nodes = {}
        self._output_shapes = {}
        _, output_channels = get_prunable_layers_in_out_channels(self._original_graph)
        self._init_pruned_layers_params(output_channels)

        self.full_flops, self.full_params_num = self._weights_flops_calc.count_flops_and_weights(
            graph, self._output_shapes
        )
        self.full_filters_num = self._weights_flops_calc.count_filters_num(graph, output_channels)

        self.current_flops, self.current_params_num = self.full_flops, self.full_params_num
        self.current_filters_num = self.full_filters_num

        self._pruned_layers_num = len(self._pruned_layer_groups_info.get_all_nodes())
        self._prunable_layers_num = len(self._original_graph.get_nodes_by_types(self._prunable_types))
        (
            self._min_possible_flops,
            self._min_possible_params,
        ) = self._calculate_flops_and_weights_in_uniformly_pruned_model(1.0)

        self._weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self._filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(
            params.get("filter_importance", PRUNING_FILTER_IMPORTANCE)
        )
        self.all_weights = params.get("all_weights", PRUNING_ALL_WEIGHTS)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get("schedule", PRUNING_SCHEDULE))
        self._scheduler = scheduler_cls(self, params)
        self._bn_adaptation = None
        self.set_pruning_level(self.pruning_init)
        self._loss = TFZeroCompressionLoss()

    @property
    def scheduler(self) -> PruningScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def compression_stage(self) -> CompressionStage:
        target_pruning_level = self.scheduler.target_level
        actual_pruning_level = self.pruning_level
        if actual_pruning_level == 0:
            return CompressionStage.UNCOMPRESSED
        if (
            isclose(actual_pruning_level, target_pruning_level, abs_tol=1e-5)
            or actual_pruning_level > target_pruning_level
        ):
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    @property
    def compression_rate(self) -> float:
        if self.prune_flops:
            return 1 - self.current_flops / self.full_flops
        return self.pruning_level

    @compression_rate.setter
    def compression_rate(self, compression_rate: float) -> None:
        is_pruning_controller_frozen = self.frozen
        self.freeze(False)
        self.set_pruning_level(compression_rate)
        self.freeze(is_pruning_controller_frozen)

    def disable_scheduler(self):
        self._scheduler = StubCompressionScheduler()
        self._scheduler.current_pruning_level = 0.0

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        if not quickly_collected_only and is_debug():
            stats = PrunedModelTheoreticalBorderline(
                self._pruned_layers_num,
                self._prunable_layers_num,
                self._min_possible_flops,
                self._min_possible_params,
                self.full_flops,
                self.full_params_num,
            )

            nncf_logger.debug(stats.to_str())

        pruned_layers_summary = self._calculate_pruned_layers_summary()
        self._update_benchmark_statistics()
        model_statistics = PrunedModelStatistics(
            self.full_flops,
            self.current_flops,
            self.full_params_num,
            self.current_params_num,
            self.full_filters_num,
            self.current_filters_num,
            pruned_layers_summary,
        )

        stats = FilterPruningStatistics(
            model_statistics, self.scheduler.current_pruning_level, self.scheduler.target_level, self.prune_flops
        )

        nncf_stats = NNCFStatistics()
        nncf_stats.register("filter_pruning", stats)
        return nncf_stats

    def freeze(self, freeze: bool = True):
        self.frozen = freeze

    def set_pruning_level(self, pruning_level: float, run_batchnorm_adaptation: bool = False):
        """
        Setup pruning masks in accordance to provided pruning level.

        :param pruning_level: Pruning level to be set.
        :param run_batchnorm_adaptation: Whether to run batchnorm adaptation after setting the pruning level.
        """
        # Pruning level from scheduler can be percentage of params that should be pruned
        self.pruning_level = pruning_level
        if not self.frozen:
            nncf_logger.info("Computing filter importance scores and binary masks...")
            if self.all_weights:
                if self.prune_flops:
                    self._set_binary_masks_for_pruned_modules_globally_by_flops_target(pruning_level)
                else:
                    self._set_binary_masks_for_pruned_layers_globally(pruning_level)
            else:
                if self.prune_flops:
                    # Looking for a layerwise pruning level needed for the required flops pruning level
                    pruning_level = self._find_uniform_pruning_level_for_target_flops(pruning_level)
                self._set_binary_masks_for_pruned_layers_groupwise(pruning_level)

        if run_batchnorm_adaptation:
            self._run_batchnorm_adaptation()

    @property
    def maximal_compression_rate(self) -> float:
        if self.prune_flops:
            return 1 - self._min_possible_flops / max(self.full_flops, 1)
        return 1.0

    def _init_pruned_layers_params(self, output_channels):
        # 1. Collect nodes output shapes
        self._output_shapes = collect_output_shapes(self._model, self._original_graph)

        # 2. Initialize next_nodes for each pruning cluster
        self._next_nodes = self._shape_pruning_proc.get_next_nodes(self._original_graph, self._pruned_layer_groups_info)

        # 3. Initialize pruning quotas
        for cluster in self._pruned_layer_groups_info.get_all_clusters():
            self._pruning_quotas[cluster.id] = floor(
                output_channels[cluster.elements[0].node_name] * self.pruning_quota
            )

    def _set_binary_masks_for_pruned_layers_groupwise(self, pruning_level: float):
        nncf_logger.debug("Setting new binary masks for pruned layers.")
        wrapped_layers = collect_wrapped_layers(self._model)

        # 0. Removing masks at the elements of the NNCFGraph
        for node in self._original_graph.topological_sort():
            node.attributes.pop("output_mask", None)

        # 1. Calculate masks
        for group in self._pruned_layer_groups_info.get_all_clusters():
            # a. Calculate the cumulative importance for all filters in the group
            cumulative_filters_importance = self._calculate_filters_importance_in_group(group)
            filters_num = len(cumulative_filters_importance)

            # b. Calculate threshold
            num_of_sparse_elems = get_rounded_pruned_element_number(
                cumulative_filters_importance.shape[0], pruning_level
            )
            threshold = sorted(cumulative_filters_importance)[min(num_of_sparse_elems, filters_num - 1)]

            # c. Initialize masks
            filter_mask = calculate_binary_mask(cumulative_filters_importance, threshold)
            for node in group.elements:
                nncf_node = self._original_graph.get_node_by_id(node.nncf_node_id)
                nncf_node.attributes["output_mask"] = TFNNCFTensor(filter_mask)

        # 2. Propagating masks across the graph
        mask_propagator = MaskPropagationAlgorithm(
            self._original_graph, TF_PRUNING_OPERATOR_METATYPES, TFNNCFPruningTensorProcessor
        )
        mask_propagator.mask_propagation()

        # 3. Apply masks to the model
        nncf_sorted_nodes = self._original_graph.topological_sort()
        for layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes if layer.name == n.layer_name][0]
            if nncf_node.attributes["output_mask"] is not None:
                self._set_operation_masks([layer], nncf_node.attributes["output_mask"].tensor)

        # Calculate actual flops and weights number with new masks
        self._update_benchmark_statistics()

    def _set_binary_masks_for_pruned_layers_globally(self, pruning_level: float):
        """
        Sets the binary mask values for layer groups according to the global pruning level.
        Filter importance scores in each group are merged into a single global list and a
        threshold value separating the pruning_level proportion of the least important filters
        in the model is calculated. Filters are pruned globally according to the threshold value.
        """
        nncf_logger.debug("Setting new binary masks for all pruned modules together.")
        filter_importances = {}
        wrapped_layers = collect_wrapped_layers(self._model)

        # 0. Remove masks at the elements of the NNCFGraph
        for node in self._original_graph.topological_sort():
            node.attributes.pop("output_mask", None)

        # 1. Calculate masks
        # a. Calculate importances for all groups of filters
        for group in self._pruned_layer_groups_info.get_all_clusters():
            cumulative_filters_importance = self._calculate_filters_importance_in_group(group)
            filter_importances[group.id] = cumulative_filters_importance

        # b. Calculate one threshold for all weights
        importances = tf.concat(list(filter_importances.values()), 0)
        threshold = sorted(importances)[int(pruning_level * importances.shape[0])]

        # c. Initialize masks
        for group in self._pruned_layer_groups_info.get_all_clusters():
            filter_mask = calculate_binary_mask(filter_importances[group.id], threshold)
            for node in group.elements:
                nncf_node = self._original_graph.get_node_by_id(node.nncf_node_id)
                nncf_node.attributes["output_mask"] = TFNNCFTensor(filter_mask)

        # 2. Propagate masks across the graph
        mask_propagator = MaskPropagationAlgorithm(
            self._original_graph, TF_PRUNING_OPERATOR_METATYPES, TFNNCFPruningTensorProcessor
        )
        mask_propagator.mask_propagation()

        # 3. Apply masks to the model
        nncf_sorted_nodes = self._original_graph.topological_sort()
        for layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes if layer.name == n.layer_name][0]
            if nncf_node.attributes["output_mask"] is not None:
                self._set_operation_masks([layer], nncf_node.attributes["output_mask"].tensor)

        # Calculate actual flops with new masks
        self._update_benchmark_statistics()

    def _set_binary_masks_for_pruned_modules_globally_by_flops_target(self, target_flops_pruning_level: float):
        """
        Prunes least important filters one-by-one until target FLOPs pruning level is achieved.
        Filters are sorted by filter importance score.
        """
        nncf_logger.debug("Setting new binary masks for pruned layers.")
        target_flops = self.full_flops * (1 - target_flops_pruning_level)
        wrapped_layers = collect_wrapped_layers(self._model)
        masks = {}

        nncf_sorted_nodes = self._original_graph.topological_sort()
        for layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes if layer.name == n.layer_name][0]
            nncf_node.attributes["output_mask"] = TFNNCFTensor(tf.ones(get_filters_num(layer)))

        # 1. Calculate importances for all groups of filters. Initialize masks.
        filter_importances = []
        group_indexes = []
        filter_indexes = []
        for group in self._pruned_layer_groups_info.get_all_clusters():
            cumulative_filters_importance = self._calculate_filters_importance_in_group(group)
            filter_importances.extend(cumulative_filters_importance)
            filters_num = len(cumulative_filters_importance)
            group_indexes.extend([group.id] * filters_num)
            filter_indexes.extend(range(filters_num))
            masks[group.id] = tf.ones(filters_num)

        # 2.
        tmp_in_channels, tmp_out_channels = get_prunable_layers_in_out_channels(self._original_graph)
        sorted_importances = sorted(zip(filter_importances, group_indexes, filter_indexes), key=lambda x: x[0])
        for _, group_id, filter_index in sorted_importances:
            if self._pruning_quotas[group_id] == 0:
                continue
            masks[group_id] = tf.tensor_scatter_nd_update(masks[group_id], [[filter_index]], [0])
            self._pruning_quotas[group_id] -= 1

            cluster = self._pruned_layer_groups_info.get_cluster_by_id(group_id)
            # Update input/output shapes of pruned elements
            self._shape_pruning_proc.prune_cluster_shapes(
                cluster=cluster,
                pruned_elems=1,
                pruning_groups_next_nodes=self._next_nodes,
                input_channels=tmp_in_channels,
                output_channels=tmp_out_channels,
            )

            flops, params_num = self._weights_flops_calc.count_flops_and_weights(
                graph=self._original_graph,
                output_shapes=self._output_shapes,
                input_channels=tmp_in_channels,
                output_channels=tmp_out_channels,
            )
            if flops <= target_flops:
                # 3. Add masks to the graph and propagate them
                for group in self._pruned_layer_groups_info.get_all_clusters():
                    for node in group.elements:
                        nncf_node = self._original_graph.get_node_by_id(node.nncf_node_id)
                        nncf_node.attributes["output_mask"] = TFNNCFTensor(masks[group.id])

                mask_propagator = MaskPropagationAlgorithm(
                    self._original_graph, TF_PRUNING_OPERATOR_METATYPES, TFNNCFPruningTensorProcessor
                )
                mask_propagator.mask_propagation()

                # 4. Set binary masks to the model
                self.current_flops = flops
                self.current_params_num = params_num
                nncf_sorted_nodes = self._original_graph.topological_sort()
                for layer in wrapped_layers:
                    nncf_node = [n for n in nncf_sorted_nodes if layer.name == n.layer_name][0]
                    if nncf_node.attributes["output_mask"] is not None:
                        self._set_operation_masks([layer], nncf_node.attributes["output_mask"].tensor)
                return
        raise nncf.InternalError(f"Unable to prune model to required flops pruning level: {target_flops_pruning_level}")

    def _set_operation_masks(self, layers: List[NNCFWrapper], filter_mask):
        for layer in layers:
            for weight_attr, ops in layer.weights_attr_ops.items():
                weight_shape = layer.layer_weights[weight_attr].shape
                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        filter_axis = get_filter_axis(layer, weight_attr)
                        broadcasted_mask = broadcast_filter_mask(filter_mask, weight_shape, filter_axis)
                        layer.ops_weights[op_name]["mask"].assign(broadcasted_mask)

    def _find_uniform_pruning_level_for_target_flops(self, target_flops_pruning_level):
        error = 0.01
        target_flops = self.full_flops * (1 - target_flops_pruning_level)
        left, right = 0.0, 1.0
        while abs(right - left) > error:
            middle = (left + right) / 2
            flops, params_num = self._calculate_flops_and_weights_in_uniformly_pruned_model(middle)
            if flops < target_flops:
                right = middle
            else:
                left = middle
        flops, params_num = self._calculate_flops_and_weights_in_uniformly_pruned_model(right)
        if flops <= target_flops:
            self.current_flops = flops
            self.current_params_num = params_num
            return right
        raise nncf.ParameterNotSupportedError(
            f"Unable to prune the model to get the required pruning level in flops = {target_flops_pruning_level}"
        )

    def _calculate_flops_and_weights_in_uniformly_pruned_model(self, pruning_level):
        (
            tmp_in_channels,
            tmp_out_channels,
        ) = self._shape_pruning_proc.calculate_in_out_channels_in_uniformly_pruned_model(
            self._original_graph,
            pruning_groups=self._pruned_layer_groups_info,
            pruning_groups_next_nodes=self._next_nodes,
            pruning_level=pruning_level,
        )

        return self._weights_flops_calc.count_flops_and_weights(
            graph=self._original_graph,
            output_shapes=self._output_shapes,
            input_channels=tmp_in_channels,
            output_channels=tmp_out_channels,
        )

    def _calculate_filters_importance_in_group(self, group: Cluster[PrunedLayerInfo]):
        """
        Calculates cumulative filters importance in the group.
        :param group: Nodes cluster
        :return a list of filter importance scores
        """
        group_layers = [self._model.get_layer(node.layer_name) for node in group.elements]
        group_filters_num = tf.constant([get_filters_num(layer) for layer in group_layers])
        filters_num = group_filters_num[0]
        assert tf.reduce_all(group_filters_num == filters_num)

        cumulative_filters_importance = tf.zeros(filters_num)
        # Calculate cumulative importance for all filters in this group
        shared_nodes: Set[str] = set()
        for minfo in group.elements:
            layer_name = minfo.layer_name
            if layer_name in shared_nodes:
                continue
            nncf_node = self._original_graph.get_node_by_id(minfo.nncf_node_id)
            if nncf_node.is_shared():
                shared_nodes.add(layer_name)
            filters_importance = self._layer_filter_importance(self._model.get_layer(layer_name))
            cumulative_filters_importance += filters_importance

        return cumulative_filters_importance

    def _update_benchmark_statistics(self):
        tmp_in_channels, tmp_out_channels = self._shape_pruning_proc.calculate_in_out_channels_by_masks(
            graph=self._original_graph,
            pruning_groups=self._pruned_layer_groups_info,
            pruning_groups_next_nodes=self._next_nodes,
            num_of_sparse_elements_by_node=self._calculate_num_of_sparse_elements_by_node(),
        )

        self.current_filters_num = self._weights_flops_calc.count_filters_num(
            graph=self._original_graph, output_channels=tmp_out_channels
        )

        self.current_flops, self.current_params_num = self._weights_flops_calc.count_flops_and_weights(
            graph=self._original_graph,
            output_shapes=self._output_shapes,
            input_channels=tmp_in_channels,
            output_channels=tmp_out_channels,
        )

    def _layer_filter_importance(self, layer: NNCFWrapper):
        layer_metatype = get_keras_layer_metatype(layer)
        if len(layer_metatype.weight_definitions) != 1:
            raise nncf.InternalError(
                f"The layer {layer.layer.name} does not support by the pruning "
                f"algorithm because it contains several weight attributes."
            )
        weight_attr = layer_metatype.weight_definitions[0].weight_attr_name
        weight = layer.layer_weights[weight_attr]
        if self.all_weights:
            weight = self._weights_normalizer(weight)
        target_weight_dim_for_compression = get_filter_axis(layer, weight_attr)
        filters_importance = self._filter_importance(weight, target_weight_dim_for_compression)
        return filters_importance

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self.config, "filter_pruning")
            )
        self._bn_adaptation.run(self.model)
