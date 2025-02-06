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
from copy import deepcopy
from typing import Set

import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.scopes import check_scopes_in_graph
from nncf.common.scopes import should_consider_scope
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.sparsity.schedulers import SparsityScheduler
from nncf.common.sparsity.statistics import LayerThreshold
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.config.extractors import extract_algo_specific_config
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.config.schemata.defaults import MAGNITUDE_SPARSITY_WEIGHT_IMPORTANCE
from nncf.config.schemata.defaults import SPARSITY_INIT
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.metatypes.tf_ops import WEIGHTABLE_TF_OP_METATYPES
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.graph.utils import get_original_name_and_instance_idx
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.sparsity.base_algorithm import SPARSITY_LAYER_METATYPES
from nncf.tensorflow.sparsity.base_algorithm import BaseSparsityController
from nncf.tensorflow.sparsity.collector import TFSparseModelStatisticsCollector
from nncf.tensorflow.sparsity.magnitude.functions import WEIGHT_IMPORTANCE_FUNCTIONS
from nncf.tensorflow.sparsity.magnitude.functions import calc_magnitude_binary_mask
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMaskWithWeightsBackup


@TF_COMPRESSION_ALGORITHMS.register("magnitude_sparsity")
class MagnitudeSparsityBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.ignored_scopes = self._algo_config.get("ignored_scopes", [])
        self._op_names = []

    def get_transformation_layout(self, model: tf.keras.Model) -> TFTransformationLayout:
        converter = TFModelConverterFactory.create(model)
        nncf_graph = converter.convert()

        check_scopes_in_graph(nncf_graph, self.ignored_scopes, self.target_scopes, self.validate_scopes)

        transformations = TFTransformationLayout()

        processed_shared_layer_names: Set[str] = set()

        for node in nncf_graph.get_all_nodes():
            if node.is_shared():
                target_layer_name, _ = get_original_name_and_instance_idx(node.node_name)
                if target_layer_name in processed_shared_layer_names:
                    continue
                processed_shared_layer_names.add(target_layer_name)

            if not should_consider_scope(node.node_name, ignored_scopes=self.ignored_scopes):
                continue

            if node.metatype in OUTPUT_NOOP_METATYPES:
                continue

            is_custom, layer_info = converter.get_layer_info_for_node(node.node_name)
            if node.metatype in SPARSITY_LAYER_METATYPES:
                # Processing a regular weighted node
                for weight_def in node.metatype.weight_definitions:
                    op_name = self._get_sparsity_operation_name(node.node_name, weight_def.weight_attr_name)
                    self._op_names.append(op_name)

                    transformations.register(
                        TFInsertionCommand(
                            target_point=TFLayerWeight(layer_info.layer_name, weight_def.weight_attr_name),
                            callable_object=BinaryMask(op_name),
                            priority=TransformationPriority.SPARSIFICATION_PRIORITY,
                        )
                    )
            elif node.metatype in WEIGHTABLE_TF_OP_METATYPES:
                assert is_custom
                # Processing a custom layer weighted node
                # Caution: here layer_name will refer to the weight itself, not to the op
                weight_attr_name = node.layer_name
                op_name = self._get_sparsity_operation_name(node.node_name, weight_attr_name)
                self._op_names.append(op_name)

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(layer_info.layer_name, weight_attr_name),
                        callable_object=BinaryMaskWithWeightsBackup(op_name, weight_attr_name),
                        priority=TransformationPriority.SPARSIFICATION_PRIORITY,
                    )
                )

        return transformations

    def _get_sparsity_operation_name(self, layer_name: str, weight_attr_name: str) -> str:
        return f"{layer_name}_{weight_attr_name}_sparsity_binary_mask"

    def _build_controller(self, model: tf.keras.Model) -> "MagnitudeSparsityController":
        """
        Simple implementation of building controller without setting builder state and loading controller's one.
        Should be called once the compressed model target_model is fully constructed.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `MagnitudeSparsityController`.
        """
        return MagnitudeSparsityController(model, self.config, self._op_names)

    def initialize(self, model: tf.keras.Model) -> None:
        pass


@api()
@ADAPTIVE_COMPRESSION_CONTROLLERS.register("tf_magnitude_sparsity")
class MagnitudeSparsityController(BaseSparsityController):
    """
    Controller class for magnitude sparsity in TF.
    """

    def __init__(self, target_model, config: NNCFConfig, op_names):
        super().__init__(target_model, op_names)
        algo_config = extract_algo_specific_config(config, "magnitude_sparsity")
        params = deepcopy(algo_config.get("params", {}))
        self._threshold = 0
        self._frozen = False
        self._weight_importance_fn = WEIGHT_IMPORTANCE_FUNCTIONS[
            params.get("weight_importance", MAGNITUDE_SPARSITY_WEIGHT_IMPORTANCE)
        ]

        sparsity_init = algo_config.get("sparsity_init", SPARSITY_INIT)
        params["sparsity_init"] = sparsity_init
        scheduler_type = params.get("schedule", "polynomial")

        if scheduler_type == "adaptive":
            raise ValueError("Magnitude sparsity algorithm do not support adaptive scheduler")

        scheduler_cls = SPARSITY_SCHEDULERS.get(scheduler_type)
        self._scheduler: SparsityScheduler = scheduler_cls(self, params)
        self._loss = TFZeroCompressionLoss()
        self._bn_adaptation = None
        self._config = config
        self.set_sparsity_level(sparsity_init)

    @property
    def scheduler(self) -> SparsityScheduler:
        return self._scheduler

    @property
    def current_sparsity_level(self) -> float:
        return self._scheduler.current_sparsity_level

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def freeze(self, freeze: bool = True):
        self._frozen = freeze

    def set_sparsity_level(self, sparsity_level, run_batchnorm_adaptation: bool = False):
        """
        Sets the sparsity level that should be applied to the model's weights.

        :param sparsity_level: Sparsity level that should be applied to the model's weights.
        :param run_batchnorm_adaptation: Whether to run batchnorm adaptation after setting the sparsity level.
        """
        if not self._frozen:
            if sparsity_level >= 1 or sparsity_level < 0:
                raise AttributeError(
                    "Sparsity level should be within interval [0,1), actual value to set is: {}".format(sparsity_level)
                )

            self._threshold = self._select_threshold(sparsity_level)
            self._set_masks_for_threshold(self._threshold)

        if run_batchnorm_adaptation:
            self._run_batchnorm_adaptation()

    def _select_threshold(self, sparsity_level):
        all_weights = self._collect_all_weights()
        if not all_weights:
            return 0.0
        all_weights_tensor = tf.sort(tf.concat(all_weights, 0))
        index = int(tf.cast(tf.size(all_weights_tensor) - 1, all_weights_tensor.dtype) * sparsity_level)
        threshold = all_weights_tensor[index].numpy()
        return threshold

    def _set_masks_for_threshold(self, threshold_val):
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                weight = wrapped_layer.layer_weights[weight_attr]

                for op_name in ops:
                    if op_name in self._op_names:
                        wrapped_layer.ops_weights[op_name]["mask"].assign(
                            calc_magnitude_binary_mask(weight, self._weight_importance_fn, threshold_val)
                        )

    def _collect_all_weights(self):
        all_weights = []
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                for op_name in ops:
                    if op_name in self._op_names:
                        all_weights.append(
                            tf.reshape(self._weight_importance_fn(wrapped_layer.layer_weights[weight_attr]), [-1])
                        )
        return all_weights

    @property
    def compression_rate(self) -> float:
        return self.statistics().magnitude_sparsity.model_statistics.sparsity_level

    @compression_rate.setter
    def compression_rate(self, compression_rate: float) -> None:
        self.freeze(False)
        self.set_sparsity_level(compression_rate)
        self.freeze(True)

    def disable_scheduler(self):
        self._scheduler = StubCompressionScheduler()
        self._scheduler.target_level = 0.0
        self._scheduler.current_sparsity_level = 0.0

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
        nncf_stats.register("magnitude_sparsity", stats)
        return nncf_stats

    def compression_stage(self) -> CompressionStage:
        if self.scheduler.current_sparsity_level >= self.scheduler.target_level:
            return CompressionStage.FULLY_COMPRESSED
        if self.scheduler.current_sparsity_level == 0:
            return CompressionStage.UNCOMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self._config, self.name)
            )
        self._bn_adaptation.run(self.model)
