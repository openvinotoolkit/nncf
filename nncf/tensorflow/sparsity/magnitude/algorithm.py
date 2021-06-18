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

from typing import Set

import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.sparsity.statistics import LayerThreshold
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import should_consider_scope
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from nncf.tensorflow.graph.converter import convert_layer_graph_to_nncf_graph
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.graph.utils import get_custom_layers
from nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from nncf.tensorflow.graph.utils import get_weight_node_name
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.sparsity.base_algorithm import BaseSparsityController
from nncf.tensorflow.sparsity.base_algorithm import SPARSITY_LAYER_METATYPES
from nncf.tensorflow.sparsity.base_algorithm import SPARSITY_TF_OP_METATYPES
from nncf.tensorflow.sparsity.collector import TFSparseModelStatisticsCollector
from nncf.tensorflow.sparsity.magnitude.functions import calc_magnitude_binary_mask
from nncf.tensorflow.sparsity.magnitude.functions import WEIGHT_IMPORTANCE_FUNCTIONS
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMaskWithWeightsBackup


@TF_COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsityBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.ignored_scopes = self.config.get('ignored_scopes', [])
        self._op_names = []

    def get_transformation_layout(self, model: tf.keras.Model) -> TFTransformationLayout:
        nncf_graph = convert_keras_model_to_nncf_graph(model)
        transformations = TFTransformationLayout()

        processed_shared_layer_names = set()  # type: Set[str]

        for node in nncf_graph.get_all_nodes():
            if node.is_shared():
                target_layer_name, _ = get_original_name_and_instance_index(node.node_name)
                if target_layer_name in processed_shared_layer_names:
                    continue
                processed_shared_layer_names.add(target_layer_name)
            else:
                target_layer_name = node.node_name

            if not (node.metatype in SPARSITY_LAYER_METATYPES and
                    should_consider_scope(node.node_name, ignored_scopes=self.ignored_scopes)):
                continue

            for weight_def in node.metatype.weight_definitions:
                op_name = self._get_sparsity_operation_name(node.node_name,
                                                            weight_def.weight_attr_name)
                self._op_names.append(op_name)

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(target_layer_name, weight_def.weight_attr_name),
                        callable_object=BinaryMask(op_name),
                        priority=TransformationPriority.SPARSIFICATION_PRIORITY
                    ))

        # Currently only applicable for models with custom layers. TODO(vshampor): expand custom layers
        # as part of building an NNCFGraph.
        for layer in get_custom_layers(model):
            layer_graph = convert_layer_graph_to_nncf_graph(layer)
            for node in layer_graph.get_all_nodes():
                if (node.metatype in SPARSITY_TF_OP_METATYPES and
                        should_consider_scope(node.node_name, ignored_scopes=self.ignored_scopes)):
                    weight_attr_name = get_weight_node_name(layer_graph, node.node_name)
                    op_name = self._get_sparsity_operation_name(node.node_name, weight_attr_name)
                    self._op_names.append(op_name)

                    transformations.register(
                        TFInsertionCommand(
                            target_point=TFLayerWeight(layer.name, weight_attr_name),
                            callable_object=BinaryMaskWithWeightsBackup(op_name, weight_attr_name),
                            priority=TransformationPriority.SPARSIFICATION_PRIORITY
                        ))

        return transformations

    def _get_sparsity_operation_name(self, layer_name: str, weight_attr_name: str) -> str:
        return f'{layer_name}_{weight_attr_name}_sparsity_binary_mask'

    def build_controller(self, model: tf.keras.Model) -> 'MagnitudeSparsityController':
        """
        Should be called once the compressed model target_model is fully constructed
        """
        return MagnitudeSparsityController(model, self.config, self._op_names)

    def initialize(self, model: tf.keras.Model) -> None:
        pass


class MagnitudeSparsityController(BaseSparsityController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """

    def __init__(self, target_model, config, op_names):
        super().__init__(target_model, op_names)
        params = config.get('params', {})
        self._threshold = 0
        self._frozen = False
        self._weight_importance_fn = WEIGHT_IMPORTANCE_FUNCTIONS[params.get('weight_importance', 'normed_abs')]

        sparsity_init = config.get('sparsity_init', 0)
        params['sparsity_init'] = sparsity_init
        scheduler_type = params.get('schedule', 'polynomial')

        if scheduler_type == 'adaptive':
            raise ValueError('Magnitude sparsity algorithm do not support adaptive scheduler')

        scheduler_cls = SPARSITY_SCHEDULERS.get(scheduler_type)
        self._scheduler = scheduler_cls(self, params)
        self._loss = TFZeroCompressionLoss()
        self._bn_adaptation = None
        self._config = config
        self.set_sparsity_level(sparsity_init)

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def freeze(self):
        self._frozen = True

    def set_sparsity_level(self, sparsity_level,
                           run_batchnorm_adaptation: bool = False):
        if not self._frozen:
            if sparsity_level >= 1 or sparsity_level < 0:
                raise AttributeError(
                    'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))

            self._threshold = self._select_threshold(sparsity_level)
            self._set_masks_for_threshold(self._threshold)

        if run_batchnorm_adaptation:
            self._run_batchnorm_adaptation()

    def _select_threshold(self, sparsity_level):
        all_weights = self._collect_all_weights()
        if not all_weights:
            return 0.0
        all_weights_tensor = tf.sort(tf.concat(all_weights, 0))
        index = int(tf.cast(tf.size(all_weights_tensor), all_weights_tensor.dtype) * sparsity_level)
        threshold = all_weights_tensor[index].numpy()
        return threshold

    def _set_masks_for_threshold(self, threshold_val):
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                weight = wrapped_layer.layer_weights[weight_attr]

                for op_name in ops:
                    if op_name in self._op_names:
                        wrapped_layer.ops_weights[op_name]['mask'].assign(
                            calc_magnitude_binary_mask(weight,
                                                       self._weight_importance_fn,
                                                       threshold_val)
                        )

    def _collect_all_weights(self):
        all_weights = []
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                for op_name in ops:
                    if op_name in self._op_names:
                        all_weights.append(tf.reshape(
                            self._weight_importance_fn(wrapped_layer.layer_weights[weight_attr]),
                            [-1]))
        return all_weights

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        collector = TFSparseModelStatisticsCollector(self.model, self._op_names)
        model_stats = collector.collect()

        threshold_stats = []
        threshold = self._select_threshold(model_stats.sparsity_level)
        for s in model_stats.sparsified_layers_summary:
            threshold_stats.append(LayerThreshold(s.name, threshold))

        target_sparsity_level = self.scheduler.current_sparsity_level

        stats = MagnitudeSparsityStatistics(model_stats, threshold_stats, target_sparsity_level)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('magnitude_sparsity', stats)
        return nncf_stats

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(**extract_bn_adaptation_init_params(self._config))
        self._bn_adaptation.run(self.model)
