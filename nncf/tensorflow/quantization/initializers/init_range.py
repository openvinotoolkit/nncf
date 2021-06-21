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

from itertools import islice
import math

import tensorflow as tf

from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.layers.data_layout import get_channel_axis
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.common.utils.progress_bar import ProgressBar
from nncf.tensorflow.quantization.initializers.collectors import MinMaxStatisticsCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanMinMaxStatisticsCollector
from nncf.tensorflow.quantization.initializers.collectors import MedianMADStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import PercentileStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanPercentileStatisticCollector


class RangeInitializer:
    def __init__(self, params: RangeInitParams):
        self.params = params
        self.dataset = params.init_range_data_loader
        self.num_steps = math.ceil(params.global_init_config.num_init_samples / self.dataset.batch_size)

        self.nncf_quantization_operation_classes = NNCF_QUANTIZATION_OPERATONS.registry_dict.values()

    def generate_stat_collector(self, per_channel: bool, channel_axes: int, input_type: str):
        range_type = self.params.global_init_config.init_type
        if range_type == 'min_max':
            return MinMaxStatisticsCollector(per_channel, channel_axes, input_type)
        if range_type == 'mean_min_max':
            return MeanMinMaxStatisticsCollector(per_channel, channel_axes, input_type)
        if range_type == 'threesigma':
            return MedianMADStatisticCollector(per_channel, channel_axes)
        if range_type == 'percentile':
            min_percentile = self.params.global_init_config.init_type_specific_params.get("min_percentile", 0.1)
            max_percentile = self.params.global_init_config.init_type_specific_params.get("max_percentile", 99.9)
            return PercentileStatisticCollector(per_channel,
                                                channel_axes,
                                                min_percentile,
                                                max_percentile)
        if range_type == 'mean_percentile':
            min_percentile = self.params.global_init_config.init_type_specific_params.get("min_percentile", 0.1)
            max_percentile = self.params.global_init_config.init_type_specific_params.get("max_percentile", 99.9)
            return MeanPercentileStatisticCollector(per_channel,
                                                    channel_axes,
                                                    input_type,
                                                    min_percentile,
                                                    max_percentile)
        raise ValueError('Range type {} is not supported.'.format(range_type))

    def run(self, model: tf.keras.Model) -> None:
        layer_statistics = []
        op_statistics = []
        handles = []
        for layer in model.layers:
            if isinstance(layer, FakeQuantize):
                channel_axes = get_channel_axis(InputType.INPUTS, '', layer)
                collector = self.generate_stat_collector(layer.per_channel, channel_axes, InputType.INPUTS)
                handles.append(layer.register_hook_pre_quantizer(collector))
                layer.enabled = False
                layer_statistics.append((layer, collector))
            elif isinstance(layer, NNCFWrapper):
                for weight_attr, ops in layer.weights_attr_ops.items():
                    for op_name, op in ops.items():
                        if op.__class__ in self.nncf_quantization_operation_classes:
                            channel_axes = get_channel_axis(InputType.WEIGHTS, weight_attr, layer)
                            collector = self.generate_stat_collector(op.per_channel, channel_axes, InputType.WEIGHTS)
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
            collector.prepare_statistics()
            layer.apply_range_initialization(collector.min, collector.max)
            layer.enabled = True

        for layer, op_name, op, collector in op_statistics:
            collector.prepare_statistics()
            weights = layer.get_operation_weights(op_name)
            op.apply_range_initialization(weights, collector.min, collector.max)
            op.enabled = True

        for handle in handles:
            handle.remove()

        for x, _ in self.dataset:
            model(x, training=False)
            break
