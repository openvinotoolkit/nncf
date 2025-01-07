# Copyright (c) 2024 Intel Corporation
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
from typing import List, Set

import numpy as np
import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.scopes import check_scopes_in_graph
from nncf.common.scopes import should_consider_scope
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.sparsity.schedulers import SparsityScheduler
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.config.extractors import extract_algo_specific_config
from nncf.config.schemata.defaults import SPARSITY_INIT
from nncf.config.schemata.defaults import SPARSITY_LEVEL_SETTING_MODE
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.graph.utils import get_original_name_and_instance_idx
from nncf.tensorflow.sparsity.base_algorithm import SPARSITY_LAYER_METATYPES
from nncf.tensorflow.sparsity.base_algorithm import BaseSparsityController
from nncf.tensorflow.sparsity.collector import TFSparseModelStatisticsCollector
from nncf.tensorflow.sparsity.rb.loss import SparseLoss
from nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight


@TF_COMPRESSION_ALGORITHMS.register("rb_sparsity")
class RBSparsityBuilder(TFCompressionAlgorithmBuilder):
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

            if not (
                node.metatype in SPARSITY_LAYER_METATYPES
                and should_consider_scope(node.node_name, ignored_scopes=self.ignored_scopes)
            ):
                continue

            _, layer_info = converter.get_layer_info_for_node(node.node_name)
            for weight_def in node.metatype.weight_definitions:
                op_name = self._get_rb_sparsity_operation_name(node.node_name, weight_def.weight_attr_name)
                self._op_names.append(op_name)

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(layer_info.layer_name, weight_def.weight_attr_name),
                        callable_object=RBSparsifyingWeight(op_name),
                        priority=TransformationPriority.SPARSIFICATION_PRIORITY,
                    )
                )

        return transformations

    def _get_rb_sparsity_operation_name(self, layer_name: str, weight_attr_name: str) -> str:
        return f"{layer_name}_{weight_attr_name}_rb_sparsity_weight"

    def _build_controller(self, model: tf.keras.Model) -> "RBSparsityController":
        """
        Simple implementation of building controller without setting builder state and loading controller's one.
        Should be called once the compressed model target_model is fully constructed.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `RBSparsityController`.
        """

        return RBSparsityController(model, self.config, self._op_names)

    def initialize(self, model: tf.keras.Model) -> None:
        pass


@api()
@ADAPTIVE_COMPRESSION_CONTROLLERS.register("tf_rb_sparsity")
class RBSparsityController(BaseSparsityController):
    """
    Controller class for regularization-based (RB) sparsity in TF.
    """

    def __init__(self, target_model, config: NNCFConfig, op_names: List[str]):
        super().__init__(target_model, op_names)
        algo_config = extract_algo_specific_config(config, "rb_sparsity")
        sparsity_init = algo_config.get("sparsity_init", SPARSITY_INIT)
        params = deepcopy(algo_config.get("params", {}))
        params["sparsity_init"] = sparsity_init
        sparsity_level_mode = params.get("sparsity_level_setting_mode", SPARSITY_LEVEL_SETTING_MODE)

        if sparsity_level_mode == "local":
            raise NotImplementedError("RB sparsity algorithm do not support local sparsity loss")

        target_ops = []
        for wrapped_layer, _, op in get_nncf_operations(self.model, self._op_names):
            target_ops.append((op, wrapped_layer.get_operation_weights(op.name)))

        self._loss = SparseLoss(target_ops)
        schedule_type = params.get("schedule", "exponential")

        if schedule_type == "adaptive":
            raise NotImplementedError("RB sparsity algorithm do not support adaptive scheduler")

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

    @property
    def current_sparsity_level(self) -> float:
        # TODO: align with torch where it currently shows the sparsity level as reported by loss object.
        #  TF does not seem to have this functionality in its SparseLoss right now.
        return self.scheduler.current_sparsity_level

    def freeze(self):
        self._loss.disable()

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        collector = TFSparseModelStatisticsCollector(self.model, self._op_names)
        model_stats = collector.collect()

        sparse_prob_sum = 0.0
        num_weights = 0
        for wrapped_layer, _, op in get_nncf_operations(self.model, self._op_names):
            operation_weights = wrapped_layer.get_operation_weights(op.name)
            mask = op.get_mask(operation_weights)
            sparse_prob_sum += tf.math.reduce_sum(tf.math.sigmoid(mask)).numpy().item()
            num_weights += np.prod(mask.shape.as_list()).item()
        mean_sparse_prob = 1.0 - (sparse_prob_sum / num_weights)

        target_sparsity_level = self.scheduler.current_sparsity_level

        stats = RBSparsityStatistics(model_stats, target_sparsity_level, mean_sparse_prob)

        nncf_stats = NNCFStatistics()
        nncf_stats.register("rb_sparsity", stats)
        return nncf_stats

    @property
    def compression_rate(self) -> float:
        return self._loss.target_sparsity_rate

    @compression_rate.setter
    def compression_rate(self, compression_rate: float) -> None:
        self.set_sparsity_level(compression_rate)

    def disable_scheduler(self):
        self._scheduler = StubCompressionScheduler()

    def compression_stage(self) -> CompressionStage:
        return None  # Issue-160174
