"""
 Copyright (c) 2022 Intel Corporation
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

import numpy as np
import tensorflow as tf

from nncf.common.quantization.initialization.range import RangeInitCollectorParams
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.utils.progress_bar import ProgressBar
from nncf.tensorflow.quantization.init_range import TFRangeInitParams
from nncf.tensorflow.quantization.init_range import RangeInitializer
from nncf.tensorflow.tensor_statistics.reduction import get_axes
from nncf.experimental.tensorflow.quantization.quantizers import InputType
from nncf.experimental.tensorflow.quantization.quantizers import NNCF_QUANTIZATION_OPERATIONS_V2
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
from nncf.tensorflow.tensor_statistics.statistics import tf_convert_stat_to_min_max_tensor_stat


class TFRangeInitParamsV2(TFRangeInitParams):
    def get_init_config_for_quantization_point_v2(self,
                                                  node_name: str,
                                                  input_type: str) -> RangeInitConfig:
        group = QuantizerGroup.WEIGHTS if input_type == InputType.WEIGHTS else QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(node_name, group)


def _get_reduction_shape(tensor_shape, channel_axes, per_channel):
    ndims = len(tensor_shape)
    channel_axes_ = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
    reduction_shape = get_axes(ndims, per_channel, channel_axes_)
    return tuple(reduction_shape)


class RangeInitializerV2(RangeInitializer):
    def __init__(self, range_init_params: TFRangeInitParamsV2):
        super().__init__(range_init_params)
        self.nncf_quantization_operation_classes = NNCF_QUANTIZATION_OPERATIONS_V2.registry_dict.values()

    def _register_op_collector(self, op, collectors, handles, op_weights):
        node_name = ''  # TODO(andrey-churkin): Use correct node_name
        init_config = self.range_init_params.get_init_config_for_quantization_point_v2(
            node_name,
            op.input_type
        )

        is_weights = op.input_type == InputType.WEIGHTS
        collector_params = RangeInitCollectorParams(is_weights, op.mode, op.per_channel)

        reduction_shape = _get_reduction_shape(op.input_shape, op.channel_axes, op.per_channel)
        if is_weights:
            num_batches = 1
        else:
            num_batches = int(np.ceil(init_config.num_init_samples / self.dataset.batch_size))

            per_sample_stats = init_config.init_type in ['mixed_min_max', 'mean_min_max']
            if collector_params.use_per_sample_stats(per_sample_stats):
                reduction_shape = reduction_shape[1:]

        collector = RangeInitializerV2.generate_stat_collector(
            reduction_shape,
            collector_params,
            init_config,
            num_batches
        )
        handles.append(op.register_hook_pre_call(collector.register_input))
        op.enabled = False
        collectors.append((op, collector, op_weights))

    def run(self, model: NNCFNetwork) -> None:
        handles = []
        collectors = []
        for op, op_weights in model.get_nncf_operations_with_params():
            if op.__class__ not in self.nncf_quantization_operation_classes:
                continue
            self._register_op_collector(op, collectors, handles, op_weights)

        for (x, _) in ProgressBar(
                islice(self.dataset, self.num_steps),
                total=self.num_steps,
                desc='Collecting tensor statistics/data'
        ):
            model(x, training=False)

        for op, collector, op_weights in collectors:
            target_stat = collector.get_statistics()
            minmax_stats = tf_convert_stat_to_min_max_tensor_stat(target_stat)

            min_values = minmax_stats.min_values
            if len(min_values.shape) != 1:
                min_values = tf.squeeze(min_values)
            max_values = minmax_stats.max_values
            if len(max_values.shape) != 1:
                max_values = tf.squeeze(max_values)

            op.apply_range_initialization(op_weights, min_values, max_values)
            op.enabled = True

        for handle in handles:
            handle.remove()

        for x, _ in self.dataset:
            model(x, training=False)
            break
