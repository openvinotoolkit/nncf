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
from tensorflow.python.keras.utils.layer_utils import count_params

from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.sparsity.schedulers import SparsityScheduler
from nncf.common.sparsity.statistics import SparsifiedLayerSummary
from nncf.common.sparsity.statistics import SparsifiedModelStatistics
from nncf.common.sparsity.statistics import RBSparsityStatistics
from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from beta.nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nxmodel
from beta.nncf.tensorflow.sparsity.base_algorithm import BaseSparsityController
from beta.nncf.tensorflow.sparsity.base_algorithm import SPARSITY_LAYERS
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss
from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from beta.nncf.tensorflow.sparsity.rb.functions import binary_mask
from beta.nncf.tensorflow.sparsity.utils import apply_fn_to_op_weights
from beta.nncf.tensorflow.utils.node import is_ignored


@TF_COMPRESSION_ALGORITHMS.register('rb_sparsity')
class RBSparsityBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.ignored_scopes = self.config.get('ignored_scopes', [])
        self._op_names = []

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_to_nxmodel(model)
        transformations = TransformationLayout()
        shared_nodes = set()

        for node_name, node in nxmodel.nodes.items():
            original_node_name, _ = get_original_name_and_instance_index(node_name)
            if (node['type'] not in SPARSITY_LAYERS or
                is_ignored(node_name, self.ignored_scopes) or
                original_node_name in shared_nodes):
                continue

            if node['is_shared']:
                shared_nodes.add(original_node_name)

            weight_attr_name = SPARSITY_LAYERS[node['type']]['weight_attr_name']
            op_name = self._get_rb_sparsity_operation_name(node_name, weight_attr_name)
            self._op_names.append(op_name)

            transformations.register(
                TFInsertionCommand(
                    target_point=TFLayerWeight(original_node_name, weight_attr_name),
                    callable_object=RBSparsifyingWeight(op_name),
                    priority=TransformationPriority.SPARSIFICATION_PRIORITY
                ))

        return transformations

    def _get_rb_sparsity_operation_name(self, layer_name, weight_attr_name):
        return f'{layer_name}_{weight_attr_name}_rb_sparsity_weight'

    def build_controller(self, model) -> BaseSparsityController:
        """
        Should be called once the compressed model target_model is fully constructed
        """

        return RBSparsityController(model, self.config, self._op_names)


class RBSparsityController(BaseSparsityController):
    def __init__(self, target_model, config, op_names: List[str]):
        super().__init__(target_model, op_names)
        sparsity_init = config.get('sparsity_init', 0)
        params = config.get('params', {})
        params['sparsity_init'] = sparsity_init
        sparsity_level_mode = params.get('sparsity_level_setting_mode', 'global')

        if sparsity_level_mode == 'local':
            raise NotImplementedError('RB sparsity algorithm do not support local sparsity loss')

        target_ops = apply_fn_to_op_weights(target_model, op_names)
        self._loss = SparseLoss(target_ops)
        schedule_type = params.get('schedule', 'exponential')

        if schedule_type == 'adaptive':
            raise NotImplementedError('RB sparsity algorithm do not support adaptive scheduler')

        scheduler_cls = SPARSITY_SCHEDULERS.get(schedule_type)
        self._scheduler = scheduler_cls(self, params)
        self.set_sparsity_level(sparsity_init)

    @property
    def scheduler(self) -> SparsityScheduler:
        return self._scheduler

    @property
    def loss(self) -> SparseLoss:
        return self._loss

    def set_sparsity_level(self, sparsity_level):
        self._loss.set_target_sparsity_loss(sparsity_level)

    def freeze(self):
        self._loss.disable()

    def statistics(self, quickly_collected_only: bool = False) -> RBSparsityStatistics:
        sparsity_levels = []
        mask_names = []
        weights_shapes = []
        weights_numbers = []
        sparse_prob_sum = tf.constant(0.)
        total_weights_number = tf.constant(0)
        total_sparsified_weights_number = tf.constant(0)
        wrapped_layers = collect_wrapped_layers(self._model)
        for wrapped_layer in wrapped_layers:
            for ops in wrapped_layer.weights_attr_ops.values():
                for op_name in ops:
                    if op_name in self._op_names:
                        mask = wrapped_layer.ops_weights[op_name]['mask']
                        sw_loss = tf.reduce_sum(binary_mask(mask))
                        weights_number = tf.size(mask)
                        sparsified_weights_number = weights_number - tf.cast(sw_loss, tf.int32)
                        mask_names.append(wrapped_layer.name + '_rb_mask')
                        weights_shapes.append(list(mask.shape))
                        weights_numbers.append(weights_number)
                        sparsity_levels.append(sparsified_weights_number / weights_number)
                        sparse_prob_sum += tf.math.reduce_sum(tf.math.sigmoid(mask))
                        total_weights_number += weights_number
                        total_sparsified_weights_number += sparsified_weights_number

        sparsity_rate_for_sparsified_modules = (total_sparsified_weights_number / total_weights_number).numpy()
        model_weights_number = count_params(self._model.weights) - total_weights_number
        sparsity_rate_for_model = (total_sparsified_weights_number / model_weights_number).numpy()
        mean_sparse_prob = (sparse_prob_sum / tf.cast(total_weights_number, tf.float32)).numpy()

        sparsity_levels = tf.keras.backend.batch_get_value(sparsity_levels)
        weights_percentages = [weights_number / total_weights_number * 100
                               for weights_number in weights_numbers]
        weights_percentages = tf.keras.backend.batch_get_value(weights_percentages)
        mask_sparsity = list(zip(mask_names, weights_shapes, sparsity_levels, weights_percentages))

        sparsified_layers_summary = []
        for mask_name, weights_shape, sparsity_level, weights_percentage in mask_sparsity:
            sparsified_layers_summary.append(
                SparsifiedLayerSummary(mask_name, weights_shape, sparsity_level, weights_percentage)
            )

        model_statistics = SparsifiedModelStatistics(sparsity_rate_for_model,
                                                     sparsity_rate_for_sparsified_modules,
                                                     sparsified_layers_summary)

        target_level = self.loss.target_sparsity_rate
        # TODO(andrey-churkin): Should be calculated when the distributed mode will be supported
        masks_consistency = 1.0

        # TODO(andrey-churkin): Check that `mean_sparse_prob` is calculated correctly
        return RBSparsityStatistics(model_statistics, masks_consistency, target_level, mean_sparse_prob)
