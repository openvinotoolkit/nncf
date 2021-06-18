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

import numpy as np
import tensorflow as tf

from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.layers.data_layout import get_channel_axis
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.layers import FakeQuantize


class MeanMinMaxStatisticsCollector:
    def __init__(self, per_channel, channel_axes):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.all_min_values = []
        self.all_max_values = []

    @property
    def min(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_min_values[0].shape:
                new_shape *= val
            for i, t in enumerate(self.all_min_values):
                self.all_min_values[i] = tf.reshape(t, shape=(new_shape))
        return tf.math.reduce_mean(tf.stack(self.all_min_values), axis=0)

    @property
    def max(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_max_values[0].shape:
                new_shape *= val
            for i, t in enumerate(self.all_max_values):
                self.all_max_values[i] = tf.reshape(t, shape=(new_shape))
        return tf.math.reduce_mean(tf.stack(self.all_max_values), axis=0)

    def call(self, inputs):
        ndims = len(inputs.shape)
        axis = list(range(ndims))
        if self.per_channel:
            for val in self.channel_axes:
                val = (ndims + val) % ndims
                axis.remove(val)
            self.all_min_values.append(tf.reduce_min(inputs, axis=axis))
            self.all_max_values.append(tf.reduce_max(inputs, axis=axis))
        else:
            axis.remove(0)
            self.all_min_values.extend(tf.unstack(tf.reduce_min(inputs, axis=axis)))
            self.all_max_values.extend(tf.unstack(tf.reduce_max(inputs, axis=axis)))

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MeanPercentileStatisticsCollector:
    def __init__(self, per_channel, channel_axes, min_percentile=5, max_percentile=95):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.all_min_values = []
        self.all_max_values = []
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def _mean_estimate_no_outliers(self, data):
        data = data.numpy()
        lower = np.percentile(data, self.min_percentile, axis=0, interpolation='nearest')
        upper = np.percentile(data, self.max_percentile, axis=0, interpolation='nearest')
        mask = np.logical_and(data >= lower, data <= upper)
        # zero out the outliers
        data_masked = data * mask
        data_masked_sum = np.sum(data_masked, axis=0)
        mask_sum = np.sum(mask, axis=0)
        mean = np.divide(data_masked_sum, mask_sum, out=np.zeros_like(data_masked_sum), where=mask_sum != 0)
        return mean

    @property
    def min(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_min_values[0].shape:
                new_shape *= val
            for i, t in enumerate(self.all_min_values):
                self.all_min_values[i] = tf.reshape(t, shape=(new_shape))
        mean = tf.py_function(self._mean_estimate_no_outliers, [tf.stack(self.all_min_values)], Tout=tf.float32)
        return mean

    @property
    def max(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_max_values[0].shape:
                new_shape *= val
            for i, t in enumerate(self.all_max_values):
                self.all_max_values[i] = tf.reshape(t, shape=(new_shape))
        mean = tf.py_function(self._mean_estimate_no_outliers, [tf.stack(self.all_max_values)], Tout=tf.float32)
        return mean

    def call(self, inputs):
        ndims = len(inputs.shape)
        axis = list(range(ndims))
        if self.per_channel:
            for val in self.channel_axes:
                val = (ndims + val) % ndims
                axis.remove(val)
            self.all_min_values.append(tf.reduce_min(inputs, axis=axis))
            self.all_max_values.append(tf.reduce_max(inputs, axis=axis))
        else:
            axis.remove(0)
            self.all_min_values.extend(tf.unstack(tf.reduce_min(inputs, axis=axis)))
            self.all_max_values.extend(tf.unstack(tf.reduce_max(inputs, axis=axis)))

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MinMaxInitializer():
    def __init__(self, params: RangeInitParams):
        self.dataset = params.init_range_data_loader

        self.num_steps = params.global_init_config.num_init_samples
        self.nncf_quantization_operation_classes = NNCF_QUANTIZATION_OPERATONS.registry_dict.values()

        range_type = params.global_init_config.init_type
        if range_type == 'mean_min_max':
            self.statistics_collector = MeanMinMaxStatisticsCollector
        elif range_type == 'mean_percentile':
            self.statistics_collector = MeanPercentileStatisticsCollector
        else:
            raise ValueError('Range type {} is not supported.'.format(range_type))

    def run(self, model):
        layer_statistics = []
        op_statistics = []
        handles = []
        for layer in model.layers:
            if isinstance(layer, FakeQuantize):
                channel_axes = get_channel_axis(InputType.INPUTS, '', layer)
                minmax = self.statistics_collector(layer.per_channel, channel_axes)
                handles.append(layer.register_hook_pre_quantizer(minmax))
                layer.enabled = False
                layer_statistics.append((layer, minmax))
            elif isinstance(layer, NNCFWrapper):
                for weight_attr, ops in layer.weights_attr_ops.items():
                    for op_name, op in ops.items():
                        if op.__class__ in self.nncf_quantization_operation_classes:
                            channel_axes = get_channel_axis(InputType.WEIGHTS, weight_attr, layer)
                            minmax = self.statistics_collector(op.per_channel, channel_axes)
                            handles.append(op.register_hook_pre_call(minmax))
                            op.enabled = False
                            op_statistics.append((layer, op_name, op, minmax))

        for x, _ in islice(self.dataset, self.num_steps):
            model(x, training=False)

        for layer, minmax in layer_statistics:
            layer.apply_minmax_initialization(minmax.min, minmax.max)
            layer.enabled = True

        for layer, op_name, op, minmax in op_statistics:
            weights = layer.get_operation_weights(op_name)
            op.apply_minmax_initialization(weights, minmax.min, minmax.max)
            op.enabled = True

        for handle in handles:
            handle.remove()

        for x, _ in self.dataset:
            model(x, training=False)
            break
