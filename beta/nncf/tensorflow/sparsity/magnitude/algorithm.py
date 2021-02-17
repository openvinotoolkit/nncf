"""
 Copyright (c) 2020 Intel Corporation
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
from tensorflow.python.keras.utils.layer_utils import count_params

from nncf.common.graph.transformations.commands import InsertionCommand
from nncf.common.graph.transformations.commands import LayerWeight
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.graph.converter import convert_layer_graph_to_nxmodel
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nxmodel
from beta.nncf.tensorflow.graph.model_transformer import TFModelTransformer
from beta.nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from beta.nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from beta.nncf.tensorflow.graph.transformations.commands import TFOperationWithWeights
from beta.nncf.tensorflow.graph.transformations.commands import TFRemovalCommand
from beta.nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.graph.utils import get_custom_layers
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.graph.utils import get_weight_node_name
from beta.nncf.tensorflow.layers.common import LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import WEIGHT_ATTR_NAME
from beta.nncf.tensorflow.sparsity.magnitude.functions import calc_magnitude_binary_mask
from beta.nncf.tensorflow.sparsity.magnitude.functions import WEIGHT_IMPORTANCE_FUNCTIONS
from beta.nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from beta.nncf.tensorflow.sparsity.magnitude.operation import BinaryMaskWithWeightsBackup
from beta.nncf.tensorflow.sparsity.utils import convert_raw_to_printable
from beta.nncf.tensorflow.sparsity.utils import strip_model_from_masks
from beta.nncf.tensorflow.utils.node import is_ignored

SPARSITY_LAYERS = LAYERS_WITH_WEIGHTS


@TF_COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsityBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.ignored_scopes = self.config.get('ignored_scopes', [])

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_to_nxmodel(model)
        transformations = TFTransformationLayout()
        shared_nodes = set()

        for node_name, node in nxmodel.nodes.items():
            original_node_name, _ = get_original_name_and_instance_index(node_name)
            if node['type'] not in SPARSITY_LAYERS \
                    or is_ignored(node_name, self.ignored_scopes) \
                    or original_node_name in shared_nodes:
                continue

            if node['is_shared']:
                shared_nodes.add(original_node_name)

            weight_attr_name = SPARSITY_LAYERS[node['type']][WEIGHT_ATTR_NAME]
            transformations.register(
                TFInsertionCommand(
                    target_point=TFLayerWeight(original_node_name, weight_attr_name),
                    callable_object=BinaryMask(),
                    priority=TransformationPriority.SPARSIFICATION_PRIORITY
                ))

        for layer in get_custom_layers(model):
            nxmodel = convert_layer_graph_to_nxmodel(layer)
            for node_name, node in nxmodel.nodes.items():
                if node['type'] in SPARSITY_LAYERS \
                        and not is_ignored(node_name, self.ignored_scopes):
                    weight_attr_name = get_weight_node_name(nxmodel, node_name)
                    transformations.register(
                        TFInsertionCommand(
                            target_point=TFLayerWeight(layer.name, weight_attr_name),
                            callable_object=BinaryMaskWithWeightsBackup(weight_attr_name),
                            priority=TransformationPriority.SPARSIFICATION_PRIORITY
                        ))

        return transformations

    def build_controller(self, model) -> TFCompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed
        """
        return MagnitudeSparsityController(model, self.config)


class MagnitudeSparsityController(TFCompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """
    def __init__(self, target_model, config):
        super().__init__(target_model)
        params = config.get('params', {})
        self.sparsity_level = self.threshold = 0
        self.frozen = False
        self.weight_importance = WEIGHT_IMPORTANCE_FUNCTIONS[params.get('weight_importance', 'normed_abs')]
        self.sparsity_init = config.get('sparsity_init', 0)
        scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
        self._scheduler = scheduler_cls(self, params)
        self.set_sparsity_level(self.sparsity_init)

    def strip_model(self, model):
        return strip_model_from_masks(model)

    def freeze(self):
        self.frozen = True

    def set_sparsity_level(self, sparsity_level):
        if not self.frozen:
            if sparsity_level >= 1 or sparsity_level < 0:
                raise AttributeError(
                    'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))
            self.sparsity_level = sparsity_level

            self.threshold = self._select_threshold()
            self._set_masks_for_threshold(self.threshold)

    def get_sparsity_init(self):
        return self.sparsity_init

    def _select_threshold(self):
        all_weights = self._collect_all_weights()
        if not all_weights:
            return 0.0
        all_weights_tensor = tf.sort(tf.concat(all_weights, 0))
        index = int(tf.cast(tf.size(all_weights_tensor), all_weights_tensor.dtype) * self.sparsity_level)
        threshold = all_weights_tensor[index].numpy()
        return threshold

    def _set_masks_for_threshold(self, threshold_val):
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                weight = wrapped_layer.layer_weights[weight_attr]

                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        wrapped_layer.ops_weights[op_name]['mask'].assign(
                            calc_magnitude_binary_mask(weight,
                                                       self.weight_importance,
                                                       threshold_val)
                        )

    def _collect_all_weights(self):
        all_weights = []
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                for op in ops.values():
                    if isinstance(op, BinaryMask):
                        all_weights.append(tf.reshape(
                            self.weight_importance(wrapped_layer.layer_weights[weight_attr]),
                            [-1]))
        return all_weights

    def statistics(self, quickly_collected_only=False):
        raw_sparsity_statistics = self.raw_statistics()
        return convert_raw_to_printable(raw_sparsity_statistics)

    def raw_statistics(self):
        raw_sparsity_statistics = {}
        sparsity_levels = []
        mask_names = []
        weights_shapes = []
        weights_numbers = []
        total_weights_number = tf.constant(0)
        total_sparsified_weights_number = tf.constant(0)
        total_bkup_weights_number = tf.constant(0)
        wrapped_layers = collect_wrapped_layers(self._model)
        for wrapped_layer in wrapped_layers:
            for ops in wrapped_layer.weights_attr_ops.values():
                for op_name, op in ops.items():
                    if isinstance(op, BinaryMaskWithWeightsBackup):
                        total_bkup_weights_number += tf.size(op.bkup_var)
                    if isinstance(op, BinaryMask):
                        mask = wrapped_layer.ops_weights[op_name]['mask']
                        mask_names.append(mask.name)
                        weights_shapes.append(list(mask.shape))
                        weights_number = tf.size(mask)
                        weights_numbers.append(weights_number)
                        sparsified_weights_number = weights_number - tf.reduce_sum(tf.cast(mask, tf.int32))
                        sparsity_levels.append(sparsified_weights_number / weights_number)
                        total_weights_number += weights_number
                        total_sparsified_weights_number += sparsified_weights_number

        sparsity_rate_for_sparsified_modules = (total_sparsified_weights_number / total_weights_number).numpy()
        model_weights_number = count_params(self._model.weights) - total_weights_number - total_bkup_weights_number
        sparsity_rate_for_model = (total_sparsified_weights_number / model_weights_number).numpy()

        raw_sparsity_statistics.update({
            'sparsity_rate_for_sparsified_modules': sparsity_rate_for_sparsified_modules,
            'sparsity_rate_for_model': sparsity_rate_for_model,
            'sparsity_threshold': self.threshold
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
