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
from typing import Set

from typing import List

import numpy as np
import tensorflow as tf

from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.sparsity.schedulers import SparsityScheduler
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from beta.nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.graph.utils import get_nncf_operations
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from beta.nncf.tensorflow.sparsity.base_algorithm import BaseSparsityController
from beta.nncf.tensorflow.sparsity.base_algorithm import SPARSITY_LAYER_METATYPES
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss
from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from beta.nncf.tensorflow.sparsity.utils import apply_fn_to_op_weights
from beta.nncf.tensorflow.sparsity.collector import TFSparseModelStatisticsCollector
from nncf.common.utils.helpers import should_consider_scope


@TF_COMPRESSION_ALGORITHMS.register('rb_sparsity')
class RBSparsityBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.ignored_scopes = self.config.get('ignored_scopes', [])
        self._op_names = []

    def get_transformation_layout(self, model):
        nncf_graph = convert_keras_model_to_nncf_graph(model)
        transformations = TransformationLayout()

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
                    should_consider_scope(node.node_name, target_scopes=None,
                                          ignored_scopes=self.ignored_scopes)):
                continue

            for weight_def in node.metatype.weight_definitions:
                op_name = self._get_rb_sparsity_operation_name(node.node_name,
                                                               weight_def.weight_attr_name)
                self._op_names.append(op_name)

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(target_layer_name, weight_def.weight_attr_name),
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

        target_level = self.loss.target_sparsity_rate
        # TODO(andrey-churkin): Should be calculated when the distributed mode will be supported
        masks_consistency = 1.0

        stats = RBSparsityStatistics(model_stats, masks_consistency, target_level, mean_sparse_prob)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('rb_sparsity', stats)
        return nncf_stats
