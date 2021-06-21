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

from math import floor
from typing import Dict, List, Set

import tensorflow as tf

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.pruning.utils import calculate_in_out_channels_in_uniformly_pruned_model
from nncf.common.pruning.utils import count_flops_and_weights
from nncf.common.pruning.utils import count_flops_and_weights_per_node
from nncf.common.pruning.utils import get_cluster_next_nodes
from nncf.common.pruning.utils import get_conv_in_out_channels
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.logger import logger as nncf_logger
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.graph.utils import get_layer_identifier
from nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from nncf.tensorflow.graph.utils import unwrap_layer
from nncf.tensorflow.layers.data_layout import get_input_channel_axis
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoBuilder
from nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoController
from nncf.tensorflow.pruning.base_algorithm import PrunedLayerInfo
from nncf.tensorflow.pruning.export_helpers import TF_PRUNING_OPERATOR_METATYPES
from nncf.tensorflow.pruning.export_helpers import TFConvolution
from nncf.tensorflow.pruning.export_helpers import TFElementwise
from nncf.tensorflow.pruning.export_helpers import TFTransposeConvolution
from nncf.tensorflow.pruning.filter_pruning.functions import calculate_binary_mask
from nncf.tensorflow.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from nncf.tensorflow.pruning.filter_pruning.functions import tensor_l2_normalizer
from nncf.tensorflow.pruning.utils import broadcast_filter_mask
from nncf.tensorflow.pruning.utils import get_filter_axis
from nncf.tensorflow.pruning.utils import get_filters_num
from nncf.tensorflow.pruning.utils import is_valid_shape
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask


@TF_COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable filter pruning during fine-tuning.
    """

    def build_controller(self, target_model: tf.keras.Model) -> TFCompressionAlgorithmController:
        return FilterPruningController(target_model,
                                       self._graph,
                                       self._op_names,
                                       self._prunable_types,
                                       self._pruned_layer_groups_info,
                                       self.config)

    def _is_pruned_layer(self, layer: tf.keras.layers.Layer) -> bool:
        # Currently prune only Convolutions
        return layer.__class__.__name__ in self._prunable_types

    def _get_op_types_of_pruned_layers(self) -> List[str]:
        return [op_name for meta_op in [TFConvolution, TFTransposeConvolution]
                for op_name in meta_op.get_all_op_aliases()]

    def _get_types_of_grouping_ops(self) -> List[str]:
        return TFElementwise.get_all_op_aliases()


class FilterPruningController(BasePruningAlgoController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable filter pruning.
    """

    def __init__(self,
                 target_model: tf.keras.Model,
                 graph: NNCFGraph,
                 op_names: List[str],
                 prunable_types: List[str],
                 pruned_layer_groups: Clusterization[PrunedLayerInfo],
                 config):
        super().__init__(target_model, op_names, prunable_types, pruned_layer_groups, config)
        self._original_graph = graph
        params = self.config.get('params', {})
        self.frozen = False
        self.pruning_quota = 0.9

        self._nodes_flops = {}  # type: Dict[NNCFNodeName, int]
        self._nodes_params_num = {}  # type: Dict[NNCFNodeName, int]
        self._layers_in_channels = {}
        self._layers_out_channels = {}
        self._layers_in_shapes = {}
        self._layers_out_shapes = {}
        self._pruning_quotas = {}
        self._next_nodes = {}
        self._init_pruned_layers_params()
        self._flops_count_init()
        self.full_flops = sum(self._nodes_flops.values())
        self.current_flops = self.full_flops
        self.full_params_num = sum(self._nodes_params_num.values())
        self.current_params_num = self.full_params_num

        self._weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self._filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(params.get('weight_importance', 'L2'))
        self.all_weights = params.get('all_weights', False)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get('schedule', 'exponential'))
        self._scheduler = scheduler_cls(self, params)
        self._bn_adaptation = None
        self.set_pruning_rate(self.pruning_init)
        self._loss = TFZeroCompressionLoss()

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        model_statistics = self._calculate_pruned_model_stats()
        self._update_benchmark_statistics()
        target_pruning_level = self.scheduler.current_pruning_level

        stats = FilterPruningStatistics(model_statistics, self.full_flops, self.current_flops,
                                        self.full_params_num, self.current_params_num, target_pruning_level)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('filter_pruning', stats)
        return nncf_stats

    def freeze(self):
        self.frozen = True

    def set_pruning_rate(self, pruning_rate: float,
                         run_batchnorm_adaptation: bool = False):
        """
        Setup pruning masks in accordance to provided pruning rate
        :param pruning_rate: pruning ration
        :return:
        """
        # Pruning rate from scheduler can be percentage of params that should be pruned
        self.pruning_rate = pruning_rate
        if not self.frozen:
            nncf_logger.info('Computing filter importances and masks...')
            if self.all_weights:
                if self.prune_flops:
                    self._set_binary_masks_for_pruned_modules_globally_by_flops_target(pruning_rate)
                else:
                    self._set_binary_masks_for_pruned_layers_globally(pruning_rate)
            else:
                if self.prune_flops:
                    # Looking for a layerwise pruning rate needed for the required flops pruning rate
                    pruning_rate = self._find_uniform_pruning_rate_for_target_flops(pruning_rate)
                self._set_binary_masks_for_pruned_layers_groupwise(pruning_rate)

        if run_batchnorm_adaptation:
            self._run_batchnorm_adaptation()

    def _init_pruned_layers_params(self):
        # 1. Initialize in/out channels for potentially prunable layers
        self._layers_in_channels, self._layers_out_channels = get_conv_in_out_channels(self._original_graph)

        # 2. Initialize next_nodes for each pruning cluster
        self._next_nodes = get_cluster_next_nodes(self._original_graph, self._pruned_layer_groups_info,
                                                  self._prunable_types)

        # 3. Initialize pruning quotas
        for cluster in self._pruned_layer_groups_info.get_all_clusters():
            self._pruning_quotas[cluster.id] = floor(self._layers_out_channels[cluster.elements[0].node_name]
                                                     * self.pruning_quota)

    def _flops_count_init(self):
        """
        Collects input/output shapes of convolutional and dense layers,
        calculates corresponding layerwise FLOPs
        """
        for node in self._original_graph.get_nodes_by_metatypes(GENERAL_CONV_LAYER_METATYPES):
            node_name, node_index = get_original_name_and_instance_index(node.node_name)
            layer = self._model.get_layer(node_name)
            layer_ = unwrap_layer(layer)

            channel_axis = get_input_channel_axis(layer)
            dims_slice = slice(channel_axis - layer_.rank, channel_axis) \
                if layer.data_format == 'channels_last' else slice(channel_axis + 1, None)
            in_shape = layer.get_input_shape_at(node_index)[dims_slice]
            out_shape = layer.get_output_shape_at(node_index)[dims_slice]

            if not is_valid_shape(in_shape) or not is_valid_shape(out_shape):
                raise RuntimeError(f'Input/output shape is not defined for layer `{layer.name}` ')

            self._layers_in_shapes[node.node_name] = in_shape
            self._layers_out_shapes[node.node_name] = out_shape

        for node in self._original_graph.get_nodes_by_metatypes(LINEAR_LAYER_METATYPES):
            node_name, node_index = get_original_name_and_instance_index(node.node_name)
            layer = self._model.get_layer(node_name)

            in_shape = layer.get_input_shape_at(node_index)[1:]
            out_shape = layer.get_output_shape_at(node_index)[1:]

            if not is_valid_shape(in_shape) or not is_valid_shape(out_shape):
                raise RuntimeError(f'Input/output shape is not defined for layer `{layer.name}` ')

            self._layers_in_shapes[node.node_name] = in_shape
            self._layers_out_shapes[node.node_name] = out_shape

        self._nodes_flops, self._nodes_params_num = \
            count_flops_and_weights_per_node(self._original_graph,
                                             self._layers_in_shapes,
                                             self._layers_out_shapes,
                                             conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                             linear_op_metatypes=LINEAR_LAYER_METATYPES)

    def _set_binary_masks_for_pruned_layers_groupwise(self, pruning_rate: float):
        nncf_logger.debug('Setting new binary masks for pruned layers.')
        wrapped_layers = collect_wrapped_layers(self._model)

        # 0. Removing masks at the elements of the NNCFGraph
        for node in self._original_graph.topological_sort():
            node.data.pop('output_mask', None)

        # 1. Calculate masks
        for group in self._pruned_layer_groups_info.get_all_clusters():
            # a. Calculate the cumulative importance for all filters in the group
            cumulative_filters_importance = self._calculate_filters_importance_in_group(group)
            filters_num = len(cumulative_filters_importance)

            # b. Calculate threshold
            num_of_sparse_elems = get_rounded_pruned_element_number(cumulative_filters_importance.shape[0],
                                                                    pruning_rate)
            threshold = sorted(cumulative_filters_importance)[min(num_of_sparse_elems, filters_num - 1)]

            # c. Initialize masks
            filter_mask = calculate_binary_mask(cumulative_filters_importance, threshold)
            for node in group.elements:
                nncf_node = self._original_graph.get_node_by_id(node.nncf_node_id)
                nncf_node.data['output_mask'] = filter_mask

        # 2. Propagating masks across the graph
        mask_propagator = MaskPropagationAlgorithm(self._original_graph, TF_PRUNING_OPERATOR_METATYPES)
        mask_propagator.mask_propagation()

        # 3. Apply masks to the model
        nncf_sorted_nodes = self._original_graph.topological_sort()
        for layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes
                         if layer.name == get_layer_identifier(n)][0]
            if nncf_node.data['output_mask'] is not None:
                self._set_operation_masks([layer], nncf_node.data['output_mask'])

        # Calculate actual flops and weights number with new masks
        self._update_benchmark_statistics()

    def _set_binary_masks_for_pruned_layers_globally(self, pruning_rate: float):
        """
        Sets the binary mask values for layer groups according to the global pruning rate.
        Filter importance scores in each group are merged into a single global list and a
        threshold value separating the pruning_rate proportion of the least important filters
        in the model is calculated. Filters are pruned globally according to the threshold value.
        """
        nncf_logger.debug('Setting new binary masks for all pruned modules together.')
        filter_importances = {}
        wrapped_layers = collect_wrapped_layers(self._model)

        # 0. Remove masks at the elements of the NNCFGraph
        for node in self._original_graph.topological_sort():
            node.data.pop('output_mask', None)

        # 1. Calculate masks
        # a. Calculate importances for all groups of filters
        for group in self._pruned_layer_groups_info.get_all_clusters():
            cumulative_filters_importance = self._calculate_filters_importance_in_group(group)
            filter_importances[group.id] = cumulative_filters_importance

        # b. Calculate one threshold for all weights
        importances = tf.concat(list(filter_importances.values()), 0)
        threshold = sorted(importances)[int(pruning_rate * importances.shape[0])]

        # c. Initialize masks
        for group in self._pruned_layer_groups_info.get_all_clusters():
            filter_mask = calculate_binary_mask(filter_importances[group.id], threshold)
            for node in group.elements:
                nncf_node = self._original_graph.get_node_by_id(node.nncf_node_id)
                nncf_node.data['output_mask'] = filter_mask

        # 2. Propagate masks across the graph
        mask_propagator = MaskPropagationAlgorithm(self._original_graph, TF_PRUNING_OPERATOR_METATYPES)
        mask_propagator.mask_propagation()

        # 3. Apply masks to the model
        nncf_sorted_nodes = self._original_graph.topological_sort()
        for layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes
                         if layer.name == get_layer_identifier(n)][0]
            if nncf_node.data['output_mask'] is not None:
                self._set_operation_masks([layer], nncf_node.data['output_mask'])

        # Calculate actual flops with new masks
        self._update_benchmark_statistics()

    def _set_binary_masks_for_pruned_modules_globally_by_flops_target(self,
                                                                      target_flops_pruning_rate: float):
        """
        Prunes least important filters one-by-one until target FLOPs pruning rate is achieved.
        Filters are sorted by filter importance score.
        """
        nncf_logger.debug('Setting new binary masks for pruned layers.')
        target_flops = self.full_flops * (1 - target_flops_pruning_rate)
        wrapped_layers = collect_wrapped_layers(self._model)
        masks = []

        nncf_sorted_nodes = self._original_graph.topological_sort()
        for layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes
                         if layer.name == get_layer_identifier(n)][0]
            nncf_node.data['output_mask'] = tf.ones(get_filters_num(layer))

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
        tmp_in_channels = self._layers_in_channels.copy()
        tmp_out_channels = self._layers_out_channels.copy()
        sorted_importances = sorted(zip(filter_importances, group_indexes, filter_indexes),
                                    key=lambda x: x[0])
        for _, group_id, filter_index in sorted_importances:
            if self._pruning_quotas[group_id] == 0:
                continue
            masks[group_id][filter_index] = 0
            self._pruning_quotas[group_id] -= 1

            # Update input/output shapes of pruned elements
            group = self._pruned_layer_groups_info.get_cluster_by_id(group_id)
            for node in group.elements:
                tmp_out_channels[node.node_name] -= 1
            for node_name in self._next_nodes[group_id]:
                tmp_in_channels[node_name] -= 1

            flops, params_num = count_flops_and_weights(self._original_graph,
                                                        self._layers_in_shapes,
                                                        self._layers_out_shapes,
                                                        input_channels=tmp_in_channels,
                                                        output_channels=tmp_out_channels,
                                                        conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                                        linear_op_metatypes=LINEAR_LAYER_METATYPES)
            if flops <= target_flops:
                # 3. Add masks to the graph and propagate them
                for group in self._pruned_layer_groups_info.get_all_clusters():
                    for node in group.elements:
                        nncf_node = self._original_graph.get_node_by_id(node.nncf_node_id)
                        nncf_node.data['output_mask'] = masks[group.id]

                mask_propagator = MaskPropagationAlgorithm(self._original_graph, TF_PRUNING_OPERATOR_METATYPES)
                mask_propagator.mask_propagation()

                # 4. Set binary masks to the model
                self.current_flops = flops
                self.current_params_num = params_num
                nncf_sorted_nodes = self._original_graph.topological_sort()
                for layer in wrapped_layers:
                    nncf_node = [n for n in nncf_sorted_nodes
                                 if layer.name == get_layer_identifier(n)][0]
                    if nncf_node.data['output_mask'] is not None:
                        self._set_operation_masks([layer], nncf_node.data['output_mask'])
                return
        raise RuntimeError(f'Unable to prune model to required flops pruning rate:'
                           f' {target_flops_pruning_rate}')

    def _set_operation_masks(self, layers: List[NNCFWrapper], filter_mask):
        for layer in layers:
            for weight_attr, ops in layer.weights_attr_ops.items():
                weight_shape = layer.layer_weights[weight_attr].shape
                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        filter_axis = get_filter_axis(layer, weight_attr)
                        broadcasted_mask = broadcast_filter_mask(filter_mask, weight_shape, filter_axis)
                        layer.ops_weights[op_name]['mask'].assign(broadcasted_mask)

    def _find_uniform_pruning_rate_for_target_flops(self, target_flops_pruning_rate):
        error = 0.01
        target_flops = self.full_flops * (1 - target_flops_pruning_rate)
        left, right = 0.0, 1.0
        while abs(right - left) > error:
            middle = (left + right) / 2
            flops, params_num = self._calculate_flops_and_weights_in_uniformly_pruned_model(middle)
            if flops < target_flops:
                right = middle
            else:
                left = middle
        flops, params_num = self._calculate_flops_and_weights_in_uniformly_pruned_model(right)
        if flops < target_flops:
            self.current_flops = flops
            self.current_params_num = params_num
            return right
        raise RuntimeError(f'Unable to prune the model to get the required '
                           f'pruning rate in flops = {target_flops_pruning_rate}')

    def _calculate_flops_and_weights_in_uniformly_pruned_model(self, pruning_rate):
        tmp_in_channels, tmp_out_channels = \
            calculate_in_out_channels_in_uniformly_pruned_model(
                pruning_groups=self._pruned_layer_groups_info.get_all_clusters(),
                pruning_rate=pruning_rate,
                full_input_channels=self._layers_in_channels,
                full_output_channels=self._layers_out_channels,
                pruning_groups_next_nodes=self._next_nodes)
        return count_flops_and_weights(self._original_graph,
                                       self._layers_in_shapes,
                                       self._layers_out_shapes,
                                       input_channels=tmp_in_channels,
                                       output_channels=tmp_out_channels,
                                       conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                       linear_op_metatypes=LINEAR_LAYER_METATYPES)

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
        shared_nodes = set()  # type: Set[str]
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

    def _calculate_flops_and_weights_pruned_model_by_masks(self):
        tmp_in_channels = self._layers_in_channels.copy()
        tmp_out_channels = self._layers_out_channels.copy()

        for group in self._pruned_layer_groups_info.get_all_clusters():
            assert all(tmp_out_channels[group.elements[0].node_name] == tmp_out_channels[node.node_name] for node in
                       group.elements)
            mask = self._original_graph.get_node_by_id(group.elements[0].nncf_node_id).data['output_mask']
            new_out_channels_num = int(sum(mask))
            num_of_sparse_elems = len(mask) - new_out_channels_num
            for node in group.elements:
                tmp_out_channels[node.node_name] = new_out_channels_num
            # Prune in_channels in all next nodes of cluster
            for node_name in self._next_nodes[group.id]:
                tmp_in_channels[node_name] -= num_of_sparse_elems

        return count_flops_and_weights(self._original_graph,
                                       self._layers_in_shapes,
                                       self._layers_out_shapes,
                                       input_channels=tmp_in_channels,
                                       output_channels=tmp_out_channels,
                                       conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                       linear_op_metatypes=LINEAR_LAYER_METATYPES)

    def _update_benchmark_statistics(self):
        self.current_flops, self.current_params_num = self._calculate_flops_and_weights_pruned_model_by_masks()

    def _layer_filter_importance(self, layer: NNCFWrapper):
        layer_metatype = get_keras_layer_metatype(layer)
        if len(layer_metatype.weight_definitions) != 1:
            raise RuntimeError(f'The layer {layer.layer.name} does not support by the pruning '
                               f'algorithm because it contains several weight attributes.')
        weight_attr = layer_metatype.weight_definitions[0].weight_attr_name
        weight = layer.layer_weights[weight_attr]
        if self.all_weights:
            weight = self._weights_normalizer(weight)
        target_weight_dim_for_compression = get_filter_axis(layer, weight_attr)
        filters_importance = self._filter_importance(weight, target_weight_dim_for_compression)
        return filters_importance

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(**extract_bn_adaptation_init_params(self.config))
        self._bn_adaptation.run(self.model)
