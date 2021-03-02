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

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.utils.layer_utils import count_params

from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.api.compression import TFCompressionScheduler
from nncf.common.graph.transformations.commands import InsertionCommand
from nncf.common.graph.transformations.commands import LayerWeight
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nxmodel
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss, SparseLossForPerLayerSparsity
from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight, OP_NAME
from beta.nncf.tensorflow.sparsity.rb.functions import binary_mask
from beta.nncf.tensorflow.sparsity.schedulers import SPARSITY_SCHEDULERS
from beta.nncf.tensorflow.sparsity.utils import convert_raw_to_printable
from beta.nncf.tensorflow.utils.node import is_ignored


PRUNING_LAYERS = {
    'Conv1D': {'weight_attr_name': 'kernel'},
    'Conv2D': {'weight_attr_name': 'kernel'},
    'DepthwiseConv2D': {'weight_attr_name': 'depthwise_kernel'},
    'Conv3D': {'weight_attr_name': 'kernel'},
    'Conv2DTranspose': {'weight_attr_name': 'kernel'},
    'Conv3DTranspose': {'weight_attr_name': 'kernel'},
    'Dense': {'weight_attr_name': 'kernel'},
    'SeparableConv1D': {'weight_attr_name': 'pointwise_kernel'},
    'SeparableConv2D': {'weight_attr_name': 'pointwise_kernel'},
    'Embedding': {'weight_attr_name': 'embeddings'},
    'LocallyConnected1D': {'weight_attr_name': 'kernel'},
    'LocallyConnected2D': {'weight_attr_name': 'kernel'}
}


@TF_COMPRESSION_ALGORITHMS.register('rb_sparsity')
class RBSparsityBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config):
        if isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy):
            raise Exception("RB sparsity algorithm don not support running in distributed mode")
        super().__init__(config)
        self.ignored_scopes = self.config.get('ignored_scopes', [])

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_to_nxmodel(model)
        transformations = TransformationLayout()
        shared_nodes = set()

        for node_name, node in nxmodel.nodes.items():
            original_node_name, _ = get_original_name_and_instance_index(node_name)
            if node['type'] not in PRUNING_LAYERS \
                    or is_ignored(node_name, self.ignored_scopes) \
                    or original_node_name in shared_nodes:
                continue

            if node['is_shared']:
                shared_nodes.add(original_node_name)

            weight_attr_name = PRUNING_LAYERS[node['type']]['weight_attr_name']
            transformations.register(
                InsertionCommand(
                    target_point=LayerWeight(original_node_name, weight_attr_name),
                    callable_object=RBSparsifyingWeight(),
                    priority=TransformationPriority.SPARSIFICATION_PRIORITY
                ))

        return transformations

    def build_controller(self, model) -> TFCompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed
        """
        # TODO: unify for TF and PyTorch
        params = self.config.get('params', {})
        if 'sparsity_init' not in params:
            params['sparsity_init'] = self.config.get("sparsity_init", 0)

        return RBSparsityController(model, params)


class RBSparsityController(TFCompressionAlgorithmController):
    def __init__(self, target_model,
                 params):
        super().__init__(target_model)
        self._scheduler = None
        self._distributed = False
        self.sparsity_init = params.get('sparsity_init', 0)
        sparsity_level_mode = params.get("sparsity_level_setting_mode", "global")
        sparsifyed_layers = collect_wrapped_layers(target_model)
        # TODO: find out purpose of this attribute
        self._check_sparsity_masks = params.get("check_sparsity_masks", False)
        if sparsity_level_mode == 'local':
            self._loss = SparseLossForPerLayerSparsity(sparsifyed_layers)
            self._scheduler = TFCompressionScheduler()
        else:
            self._loss = SparseLoss(sparsifyed_layers)  # type: SparseLoss
            schedule_type = params.get("schedule", "exponential")
            scheduler_cls = SPARSITY_SCHEDULERS.get(schedule_type)
            self._scheduler = scheduler_cls(self, params)

    def set_sparsity_level(self, sparsity_level, target_layer: NNCFWrapper = None):
        if target_layer is None:
            #pylint:disable=no-value-for-parameter
            self._loss.set_target_sparsity_loss(sparsity_level)
        else:
            self._loss.set_target_sparsity_loss(sparsity_level, target_layer)

    def freeze(self):
        self._loss.disable()

    def statistics(self):
        raw_sparsity_statistics = self.raw_statistics()
        return convert_raw_to_printable(raw_sparsity_statistics)

    def raw_statistics(self):
        raw_sparsity_statistics = {}
        sparsity_levels = []
        mask_names = []
        weights_shapes = []
        weights_numbers = []
        sparse_prob_sum = tf.constant(0.)
        total_weights_number = tf.constant(0)
        total_sparsified_weights_number = tf.constant(0)
        wrapped_layers = collect_wrapped_layers(self._model)
        for wrapped_layer in wrapped_layers:
            mask = wrapped_layer.ops_weights[OP_NAME]['mask']
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

        raw_sparsity_statistics.update({
            'sparsity_rate_for_sparsified_modules': sparsity_rate_for_sparsified_modules,
            'sparsity_rate_for_model': sparsity_rate_for_model,
            'mean_sparse_prob': mean_sparse_prob,
            'target_sparsity_rate': self.loss.target_sparsity_rate,
        })

        sparsity_levels = tf.keras.backend.batch_get_value(sparsity_levels)
        weights_percentages = [weights_number / total_weights_number * 100
                               for weights_number in weights_numbers]
        weights_percentages = tf.keras.backend.batch_get_value(weights_percentages)
        mask_sparsity = list(zip(mask_names, weights_shapes, sparsity_levels, weights_percentages))
        raw_sparsity_statistics['sparsity_statistic_by_module'] = []
        for mask_name, weights_shape, sparsity_level, weights_percentage in mask_sparsity:
            raw_sparsity_statistics['sparsity_statistic_by_module'].append({
                'Name': mask_name,
                'Weight\'s Shape': weights_shape,
                'SR': sparsity_level,
                '% weights': weights_percentage
            })

        return raw_sparsity_statistics

    def get_compression_metrics(self, model_loss):
        return [
            model_loss,
            tfa.metrics.MeanMetricWrapper(self.loss,
                                          name='rb_loss')
            ]

    def get_sparsity_init(self):
        return self.sparsity_init
