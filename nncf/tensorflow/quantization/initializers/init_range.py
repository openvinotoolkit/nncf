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
from copy import deepcopy
from itertools import islice
import math

import tensorflow as tf

from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.utils.progress_bar import ProgressBar
from nncf.common.utils.helpers import should_consider_scope
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATIONS
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.layers.data_layout import get_channel_axis
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.tensorflow.quantization.initializers.collectors import MinMaxStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MixedMinMaxStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanMinMaxStatisticsCollector
from nncf.tensorflow.quantization.initializers.collectors import MedianMADStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import PercentileStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanPercentileStatisticCollector


class TFRangeInitParams(RangeInitParams):
    def get_max_num_init_steps(self) -> int:
        steps = []
        if self.global_init_config is not None:
            steps.append(self.global_init_config.num_init_samples)
        for pl_config in self.per_layer_range_init_configs:
            steps.append(pl_config.num_init_samples)
        batch_size = self.init_range_data_loader.batch_size
        return math.ceil(max(steps) / batch_size)

    def get_init_config_for_quantization_point(self, layer: tf.keras.layers.Layer,
                                               input_type: str) -> RangeInitConfig:
        if input_type == InputType.WEIGHTS:
            node_name = layer.name
            group = QuantizerGroup.WEIGHTS
        else:
            node_name = layer.name.replace('/fake_quantize', '')
            group = QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(node_name, group)

    def get_init_config_for_scope_and_group(self, node_name: str, group: QuantizerGroup) -> RangeInitConfig:
        matches = []  # type: List[RangeInitConfig]
        for pl_config in self.per_layer_range_init_configs:
            if should_consider_scope(node_name,
                                     ignored_scopes=pl_config.ignored_scopes,
                                     target_scopes=pl_config.target_scopes):
                if group == pl_config.target_group or pl_config.target_group is None:
                    matches.append(RangeInitConfig(pl_config.init_type, pl_config.num_init_samples,
                                                   pl_config.init_type_specific_params))
        if len(matches) > 1:
            raise ValueError('Location {} matches more than one per-layer initialization parameter '
                             'definition!'.format(str(node_name)))
        if len(matches) == 1:
            return matches[0]
        if not matches and self.global_init_config is not None:
            return deepcopy(self.global_init_config)

        raise ValueError('Location {} does not match any per-layer initialization parameter '
                         'definition!'.format(str(node_name)))


class RangeInitializer:
    def __init__(self, range_init_params: TFRangeInitParams):
        self.range_init_params = range_init_params
        self.dataset = range_init_params.init_range_data_loader
        self.num_steps = range_init_params.get_max_num_init_steps()

        self.nncf_quantization_operation_classes = NNCF_QUANTIZATION_OPERATIONS.registry_dict.values()

    @staticmethod
    def generate_stat_collector(init_config, per_channel: bool, channel_axes: int, input_type: str, mode: str):
        range_type = init_config.init_type
        num_samples = init_config.num_init_samples
        if range_type == 'min_max':
            return MinMaxStatisticCollector(per_channel, channel_axes, input_type, mode, num_samples)
        if range_type == 'mixed_min_max':
            return MixedMinMaxStatisticCollector(per_channel, channel_axes, input_type, mode, num_samples)
        if range_type == 'mean_min_max':
            return MeanMinMaxStatisticsCollector(per_channel, channel_axes, input_type, mode, num_samples)
        if range_type == 'threesigma':
            return MedianMADStatisticCollector(per_channel, channel_axes, input_type, num_samples)
        if range_type == 'percentile':
            min_percentile = init_config.init_type_specific_params.get('min_percentile', 0.1)
            max_percentile = init_config.init_type_specific_params.get('max_percentile', 99.9)
            return PercentileStatisticCollector(per_channel, channel_axes, input_type,
                                                [min_percentile, max_percentile], num_samples)
        if range_type == 'mean_percentile':
            min_percentile = init_config.init_type_specific_params.get('min_percentile', 0.1)
            max_percentile = init_config.init_type_specific_params.get('max_percentile', 99.9)
            return MeanPercentileStatisticCollector(per_channel, channel_axes, input_type,
                                                    [min_percentile, max_percentile], num_samples)
        raise ValueError(f'Range type {range_type} is not supported.')

    def run(self, model: tf.keras.Model) -> None:
        layer_statistics = []
        op_statistics = []
        handles = []
        for layer in model.layers:
            if isinstance(layer, FakeQuantize):
                channel_axes = get_channel_axis(InputType.INPUTS, '', layer)
                init_config = self.range_init_params.get_init_config_for_quantization_point(layer, InputType.INPUTS)
                collector = RangeInitializer.generate_stat_collector(init_config, layer.per_channel,
                                                                     channel_axes, InputType.INPUTS, layer.mode)
                handles.append(layer.register_hook_pre_quantizer(collector))
                layer.enabled = False
                layer_statistics.append((layer, collector))
            elif isinstance(layer, NNCFWrapper):
                for weight_attr, ops in layer.weights_attr_ops.items():
                    for op_name, op in ops.items():
                        if op.__class__ in self.nncf_quantization_operation_classes:
                            channel_axes = get_channel_axis(InputType.WEIGHTS, weight_attr, layer)
                            init_config = self.range_init_params.\
                                get_init_config_for_quantization_point(layer, InputType.WEIGHTS)
                            collector = RangeInitializer.generate_stat_collector(init_config, op.per_channel,
                                                                                 channel_axes, InputType.WEIGHTS,
                                                                                 op.mode)
                            handles.append(op.register_hook_pre_call(collector))
                            op.enabled = False
                            op_statistics.append((layer, op_name, op, collector))

        for (x, _) in ProgressBar(
                islice(self.dataset, self.num_steps),
                total=self.num_steps,
                desc='Collecting tensor statistics/data'
        ):
            model(x, training=False)

        for layer, collector in layer_statistics:
            target_stat = collector.get_statistics()
            minmax_stats = MinMaxTensorStatistic.from_stat(target_stat)
            layer.apply_range_initialization(minmax_stats.min_values, minmax_stats.max_values)
            layer.enabled = True

        for layer, op_name, op, collector in op_statistics:
            weights = layer.get_operation_weights(op_name)
            target_stat = collector.get_statistics()
            minmax_stats = MinMaxTensorStatistic.from_stat(target_stat)
            op.apply_range_initialization(weights, minmax_stats.min_values, minmax_stats.max_values)
            op.enabled = True

        for handle in handles:
            handle.remove()

        for x, _ in self.dataset:
            model(x, training=False)
            break
