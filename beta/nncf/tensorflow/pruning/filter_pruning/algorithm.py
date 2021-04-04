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
from typing import List

import tensorflow as tf

from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.api.compression import TFZeroCompressionLoss
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from beta.nncf.tensorflow.layers.common import LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import WEIGHT_ATTR_NAME
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoBuilder
from beta.nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoController
from beta.nncf.tensorflow.pruning.export_helpers import TFElementwise
from beta.nncf.tensorflow.pruning.export_helpers import TFConvolution
from beta.nncf.tensorflow.pruning.export_helpers import TFTransposeConvolution
from beta.nncf.tensorflow.pruning.filter_pruning.functions import calculate_binary_mask
from beta.nncf.tensorflow.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from beta.nncf.tensorflow.pruning.filter_pruning.functions import tensor_l2_normalizer
from beta.nncf.tensorflow.pruning.utils import broadcast_filter_mask
from beta.nncf.tensorflow.pruning.utils import get_filter_axis
from beta.nncf.tensorflow.pruning.utils import get_filters_num
from nncf.common.pruning.model_analysis import Clusterization
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.pruning.schedulers import PRUNING_SCHEDULERS
<<<<<<< c586e89804be488f825b18e93ff83ff2ce06883e
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from beta.nncf.tensorflow.pruning.export_helpers import TF_PRUNING_OPERATOR_METATYPES
=======
from nncf.api.compression import CompressionScheduler
<<<<<<< 1844b17660664362944ede6ffae2722512ff7f60
>>>>>>> Fix test auxiliary struct. Add type hints.
=======
from nncf.api.compression import CompressionLoss

>>>>>>> After rebase.


@TF_COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable filter pruning during fine-tuning.
    """

    def build_controller(self, target_model: tf.keras.Model) -> TFCompressionAlgorithmController:
        return FilterPruningController(target_model,
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
                 op_names: List[str],
                 prunable_types: List[str],
                 pruned_layer_groups: Clusterization,
                 config):
        super().__init__(target_model, op_names, prunable_types, pruned_layer_groups, config)
        params = self.config.get('params', {})
        self.frozen = False

        self._weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self._filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(params.get('weight_importance', 'L2'))
        self.all_weights = params.get('all_weights', False)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get('schedule', 'exponential'))
        self._scheduler = scheduler_cls(self, params)
<<<<<<< 1844b17660664362944ede6ffae2722512ff7f60
        self.set_pruning_rate(self.pruning_init)
=======
        self._loss = TFZeroCompressionLoss()
>>>>>>> After rebase.

    def freeze(self):
        self.frozen = True

    def set_pruning_rate(self, pruning_rate: float):
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
                self._set_binary_masks_for_all_pruned_layers(pruning_rate)
            else:
                self._set_binary_masks_for_filters(pruning_rate)

    def _set_binary_masks_for_filters(self, pruning_rate: float):
        nncf_logger.debug('Setting new binary masks for pruned layers.')
        wrapped_layers = collect_wrapped_layers(self._model)
        graph = convert_keras_model_to_nncf_graph(self._model)
        nncf_sorted_nodes = graph.topological_sort()

        # 0. Removing masks at the nodes of the graph
        for node in nncf_sorted_nodes:
            node.data.pop('output_mask', None)

        # 1. Calculate masks
        for group in self._pruned_layer_groups_info.get_all_clusters():
            group_layer_names = [node.layer_name for node in group.nodes]
            group_filters_num = tf.constant([get_filters_num(wrapped_layer)
                                             for wrapped_layer in wrapped_layers
                                             if wrapped_layer.layer.name in group_layer_names])
            filters_num = group_filters_num[0]
            assert tf.reduce_all(group_filters_num == filters_num)

            layers = []
            cumulative_filters_importance = tf.zeros(filters_num)

            # a. Calculate cumulative importance for all filters in group
            for minfo in group.nodes:
                layer = [layer for layer in wrapped_layers if layer.layer.name == minfo.layer_name][0]
                layers.append(layer)
                filters_importance = self._layer_filter_importance(layer)
                cumulative_filters_importance += filters_importance

            # b. Calculate threshold
            num_of_sparse_elems = get_rounded_pruned_element_number(cumulative_filters_importance.shape[0],
                                                                    pruning_rate)
            threshold = sorted(cumulative_filters_importance)[min(num_of_sparse_elems, filters_num - 1)]
            filter_mask = calculate_binary_mask(cumulative_filters_importance, threshold)

            for layer in layers:
                nncf_node = [n for n in nncf_sorted_nodes if layer.name == n.data['original_name']][0]
                nncf_node.data['output_mask'] = filter_mask

        # 2. Propagating masks across the graph
        mask_propagator = MaskPropagationAlgorithm(graph, TF_PRUNING_OPERATOR_METATYPES)
        mask_propagator.mask_propagation()

        # 3. Apply the masks
        for wrapped_layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes if wrapped_layer.name == n.data['original_name']][0]
            if nncf_node.data['output_mask'] is not None:
                self._set_operation_masks([wrapped_layer], nncf_node.data['output_mask'])

    def _layer_filter_importance(self, layer: NNCFWrapper):
        layer_type = layer.layer.__class__.__name__
        weight_attr = LAYERS_WITH_WEIGHTS[layer_type][WEIGHT_ATTR_NAME]
        weight = layer.layer_weights[weight_attr]
        if self.all_weights:
            weight = self._weights_normalizer(weight)
        target_weight_dim_for_compression = get_filter_axis(layer, weight_attr)
        filters_importance = self._filter_importance(weight, target_weight_dim_for_compression)
        return filters_importance

    def _set_operation_masks(self, layers: List[NNCFWrapper], filter_mask):
        for layer in layers:
            for weight_attr, ops in layer.weights_attr_ops.items():
                weight_shape = layer.layer_weights[weight_attr].shape
                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        filter_axis = get_filter_axis(layer, weight_attr)
                        broadcasted_mask = broadcast_filter_mask(filter_mask, weight_shape, filter_axis)
                        layer.ops_weights[op_name]['mask'].assign(broadcasted_mask)

    def _set_binary_masks_for_all_pruned_layers(self, pruning_rate: float):
        nncf_logger.debug('Setting new binary masks for all pruned modules together.')
        filter_importances = []
        layers = []

        wrapped_layers = collect_wrapped_layers(self._model)
        graph = convert_keras_model_to_nncf_graph(self._model)
        nncf_sorted_nodes = graph.topological_sort()

        # 0. Removing masks at the nodes of the graph
        for node in nncf_sorted_nodes:
            node.data.pop('output_mask', None)

        # 1. Calculate importances for all groups of  filters
        for group in self._pruned_layer_groups_info.get_all_clusters():
            group_layer_names = [node.layer_name for node in group.nodes]
            group_filters_num = tf.constant([get_filters_num(wrapped_layer)
                                             for wrapped_layer in wrapped_layers
                                             if wrapped_layer.layer.name in group_layer_names])
            filters_num = group_filters_num[0]
            assert tf.reduce_all(group_filters_num == filters_num)

            group_layers = []
            cumulative_filters_importance = tf.zeros(filters_num)
            # Calculate cumulative importance for all filters in this group
            for minfo in group.nodes:
                layer = [layer for layer in wrapped_layers if layer.layer.name == minfo.layer_name][0]
                group_layers.append(layer)
                filters_importance = self._layer_filter_importance(layer)
                cumulative_filters_importance += filters_importance

            layers.append(group_layers)
            filter_importances.append(cumulative_filters_importance)

        # 2. Calculate one threshold for all weights
        importances = tf.concat(filter_importances, 0)
        threshold = sorted(importances)[int(pruning_rate * importances.shape[0])]

        # 3. Set binary masks for filters in grops
        for i, group in enumerate(self._pruned_layer_groups_info.get_all_clusters()):
            filter_mask = calculate_binary_mask(filter_importances[i], threshold)
            for layer in layers[i]:
                nncf_node = [n for n in nncf_sorted_nodes if layer.name == n.node_name][0]
                nncf_node.data['output_mask'] = filter_mask

        # 2. Propagating masks across the graph
        mask_propagator = MaskPropagationAlgorithm(graph, TF_PRUNING_OPERATOR_METATYPES)
        mask_propagator.mask_propagation()

        # 3. Apply the masks
        for wrapped_layer in wrapped_layers:
            nncf_node = [n for n in nncf_sorted_nodes if wrapped_layer.name == n.node_name][0]
            if nncf_node.data['output_mask'] is not None:
                self._set_operation_masks([wrapped_layer], nncf_node.data['output_mask'])

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
<<<<<<< 1844b17660664362944ede6ffae2722512ff7f60
    def loss(self):
        pass
=======
    def loss(self) -> CompressionLoss:
        return self._loss
>>>>>>> After rebase.
