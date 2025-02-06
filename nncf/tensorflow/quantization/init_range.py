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

import math
from copy import deepcopy
from itertools import islice
from typing import List

import numpy as np
import tensorflow as tf

from nncf.common.logging.progress_bar import ProgressBar
from nncf.common.quantization.initialization.range import RangeInitCollectorParams
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.config.schemata.defaults import MAX_PERCENTILE
from nncf.config.schemata.defaults import MIN_PERCENTILE
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATIONS
from nncf.tensorflow.layers.data_layout import get_channel_axis
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.tensorflow.tensor_statistics.collectors import TFMeanMinMaxStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMeanPercentileStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMedianMADStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMinMaxStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMixedMinMaxStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFPercentileStatisticCollector
from nncf.tensorflow.tensor_statistics.reduction import get_reduction_shape_activations
from nncf.tensorflow.tensor_statistics.reduction import get_reduction_shape_weights
from nncf.tensorflow.tensor_statistics.statistics import tf_convert_stat_to_min_max_tensor_stat


class TFRangeInitParams(RangeInitParams):
    def get_max_num_init_steps(self) -> int:
        steps = []
        if self.global_init_config is not None:
            steps.append(self.global_init_config.num_init_samples)
        for pl_config in self.per_layer_range_init_configs:
            steps.append(pl_config.num_init_samples)
        batch_size = self.init_range_data_loader.batch_size
        return math.ceil(max(steps) / batch_size)

    def get_init_config_for_quantization_point(self, layer: tf.keras.layers.Layer, input_type: str) -> RangeInitConfig:
        if input_type == InputType.WEIGHTS:
            node_name = layer.name
            group = QuantizerGroup.WEIGHTS
        else:
            node_name = layer.name.replace("/fake_quantize", "")
            group = QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(node_name, group)

    def get_init_config_for_scope_and_group(self, node_name: str, group: QuantizerGroup) -> RangeInitConfig:
        matches: List[RangeInitConfig] = []
        for pl_config in self.per_layer_range_init_configs:
            should_be_considered = should_consider_scope(
                node_name, ignored_scopes=pl_config.ignored_scopes, target_scopes=pl_config.target_scopes
            )
            if should_be_considered and (group == pl_config.target_group or pl_config.target_group is None):
                matches.append(
                    RangeInitConfig(
                        pl_config.init_type, pl_config.num_init_samples, pl_config.init_type_specific_params
                    )
                )
        if len(matches) > 1:
            raise ValueError(
                "Location {} matches more than one per-layer initialization parameter "
                "definition!".format(str(node_name))
            )
        if len(matches) == 1:
            return matches[0]
        if not matches and self.global_init_config is not None:
            return deepcopy(self.global_init_config)

        raise ValueError(
            "Location {} does not match any per-layer initialization parameter definition!".format(str(node_name))
        )


class RangeInitializer:
    def __init__(self, range_init_params: TFRangeInitParams):
        self.range_init_params = range_init_params
        self.dataset = range_init_params.init_range_data_loader
        self.num_steps = range_init_params.get_max_num_init_steps()

        self.nncf_quantization_operation_classes = NNCF_QUANTIZATION_OPERATIONS.registry_dict.values()

    @staticmethod
    def generate_stat_collector(
        reduction_shape: ReductionAxes,
        collector_params: RangeInitCollectorParams,
        init_config: RangeInitConfig,
        num_samples_to_collect_override: int = None,
    ) -> TensorStatisticCollectorBase:
        range_type = init_config.init_type
        num_samples = init_config.num_init_samples
        if num_samples_to_collect_override is not None:
            num_samples = num_samples_to_collect_override

        if range_type == "min_max":
            return TFMinMaxStatisticCollector(collector_params.use_abs_max, reduction_shape, num_samples)
        if range_type == "mixed_min_max":
            return TFMixedMinMaxStatisticCollector(
                collector_params.use_per_sample_stats(per_sample_stats=True),
                collector_params.use_abs_max,
                collector_params.use_means_of_mins,
                collector_params.use_means_of_maxs,
                reduction_shape,
                num_samples,
            )
        if range_type == "mean_min_max":
            return TFMeanMinMaxStatisticCollector(
                collector_params.use_per_sample_stats(per_sample_stats=True),
                collector_params.use_abs_max,
                reduction_shape,
                num_samples,
            )
        if range_type == "threesigma":
            return TFMedianMADStatisticCollector(reduction_shape, num_samples)
        if range_type == "percentile":
            min_percentile = init_config.init_type_specific_params.get("min_percentile", MIN_PERCENTILE)
            max_percentile = init_config.init_type_specific_params.get("max_percentile", MAX_PERCENTILE)
            return TFPercentileStatisticCollector([min_percentile, max_percentile], reduction_shape, num_samples)
        if range_type == "mean_percentile":
            min_percentile = init_config.init_type_specific_params.get("min_percentile", MIN_PERCENTILE)
            max_percentile = init_config.init_type_specific_params.get("max_percentile", MAX_PERCENTILE)
            return TFMeanPercentileStatisticCollector([min_percentile, max_percentile], reduction_shape, num_samples)
        raise ValueError(f"Range type {range_type} is not supported.")

    def _register_layer_statistics(self, layer: tf.keras.layers.Layer, layer_statistics: list, handles: list):
        channel_axes = get_channel_axis(InputType.INPUTS, "", layer)
        init_config = self.range_init_params.get_init_config_for_quantization_point(layer, InputType.INPUTS)

        is_weights = False
        collector_params = RangeInitCollectorParams(is_weights, layer.mode, layer.per_channel)
        per_sample_stats = init_config.init_type in ["mixed_min_max", "mean_min_max"]

        reduction_shape = get_reduction_shape_activations(
            layer, channel_axes, collector_params.use_per_sample_stats(per_sample_stats)
        )

        num_batches = int(np.ceil(init_config.num_init_samples / self.dataset.batch_size))

        collector = RangeInitializer.generate_stat_collector(
            reduction_shape, collector_params, init_config, num_batches
        )
        handles.append(layer.register_hook_pre_quantizer(collector.register_input))
        layer.enabled = False
        layer_statistics.append((layer, collector))

    def _register_op_statistics(self, layer: tf.keras.layers.Layer, op_statistics: list, handles: list):
        for weight_attr, ops in layer.weights_attr_ops.items():
            for op_name, op in ops.items():
                if op.__class__ in self.nncf_quantization_operation_classes:
                    channel_axes = get_channel_axis(InputType.WEIGHTS, weight_attr, layer)
                    init_config = self.range_init_params.get_init_config_for_quantization_point(
                        layer, InputType.WEIGHTS
                    )

                    is_weights = True
                    collector_params = RangeInitCollectorParams(is_weights, op.mode, op.per_channel)

                    reduction_shape = get_reduction_shape_weights(layer, weight_attr, channel_axes, op.per_channel)

                    # No need to store extra statistics in memory since weights won't change during range init
                    num_batches = 1

                    collector = RangeInitializer.generate_stat_collector(
                        reduction_shape, collector_params, init_config, num_batches
                    )
                    handles.append(op.register_hook_pre_call(collector.register_input))
                    op.enabled = False
                    op_statistics.append((layer, op_name, op, collector))

    def run(self, model: tf.keras.Model) -> None:
        layer_statistics = []
        op_statistics = []
        handles = []
        for layer in model.layers:
            if isinstance(layer, FakeQuantize):
                self._register_layer_statistics(layer, layer_statistics, handles)
            elif isinstance(layer, NNCFWrapper):
                self._register_op_statistics(layer, op_statistics, handles)

        for x, _ in ProgressBar(
            islice(self.dataset, self.num_steps), total=self.num_steps, desc="Collecting tensor statistics/data"
        ):
            model(x, training=False)

        for layer, collector in layer_statistics:
            target_stat = collector.get_statistics()
            minmax_stats = tf_convert_stat_to_min_max_tensor_stat(target_stat)
            layer.apply_range_initialization(tf.squeeze(minmax_stats.min_values), tf.squeeze(minmax_stats.max_values))
            layer.enabled = True

        for layer, op_name, op, collector in op_statistics:
            weights = layer.get_operation_weights(op_name)
            target_stat = collector.get_statistics()
            minmax_stats = tf_convert_stat_to_min_max_tensor_stat(target_stat)
            min_values = minmax_stats.min_values
            if len(min_values.shape) != 1:
                min_values = tf.squeeze(min_values)
            max_values = minmax_stats.max_values
            if len(max_values.shape) != 1:
                max_values = tf.squeeze(max_values)
            op.apply_range_initialization(weights, min_values, max_values)
            op.enabled = True

        for handle in handles:
            handle.remove()

        for x, _ in self.dataset:
            model(x, training=False)
            break
