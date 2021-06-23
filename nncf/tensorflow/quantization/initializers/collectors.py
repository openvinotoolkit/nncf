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

import numpy as np
import tensorflow as tf

from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.initializers.utils import get_per_channel_history
from nncf.tensorflow.quantization.initializers.utils import discard_zeros
from nncf.tensorflow.quantization.initializers.utils import get_axes


class MinMaxStatisticsCollector:
    def __init__(self, per_channel: bool, channel_axes: int, input_type: str):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.input_type = input_type
        self.all_min_values = []
        self.all_max_values = []

    @property
    def min(self) -> tf.Tensor:
        return tf.math.reduce_min(tf.stack(self.all_min_values), axis=0)

    @property
    def max(self) -> tf.Tensor:
        return tf.math.reduce_max(tf.stack(self.all_max_values), axis=0)

    def prepare_statistics(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_min_values[0].shape:
                new_shape *= val
            for i, _ in enumerate(self.all_min_values):
                self.all_min_values[i] = tf.reshape(self.all_min_values[i], shape=(new_shape))
                self.all_max_values[i] = tf.reshape(self.all_max_values[i], shape=(new_shape))

    def call(self, inputs: tf.Tensor):
        ndims = len(inputs.shape)
        axis = get_axes(ndims, self.per_channel, self.channel_axes)

        if self.input_type == InputType.INPUTS:
            axis.remove(0)
            self.all_min_values.extend(tf.unstack(tf.reduce_min(inputs, axis=axis)))
            self.all_max_values.extend(tf.unstack(tf.reduce_max(inputs, axis=axis)))
        elif self.input_type == InputType.WEIGHTS:
            self.all_min_values.append(tf.reduce_min(inputs, axis=axis))
            self.all_max_values.append(tf.reduce_max(inputs, axis=axis))

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MeanMinMaxStatisticsCollector:
    def __init__(self, per_channel: bool, channel_axes: int, input_type: str):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.input_type = input_type
        self.all_min_values = []
        self.all_max_values = []

    @property
    def min(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self.all_min_values), axis=0)

    @property
    def max(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self.all_max_values), axis=0)

    def prepare_statistics(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_min_values[0].shape:
                new_shape *= val
            for i, _ in enumerate(self.all_min_values):
                self.all_min_values[i] = tf.reshape(self.all_min_values[i], shape=(new_shape))
                self.all_max_values[i] = tf.reshape(self.all_max_values[i], shape=(new_shape))

    def call(self, inputs: tf.Tensor):
        ndims = len(inputs.shape)
        axis = get_axes(ndims, self.per_channel, self.channel_axes)

        if self.input_type == InputType.INPUTS:
            axis.remove(0)
            self.all_min_values.extend(tf.unstack(tf.reduce_min(inputs, axis=axis)))
            self.all_max_values.extend(tf.unstack(tf.reduce_max(inputs, axis=axis)))
        elif self.input_type == InputType.WEIGHTS:
            self.all_min_values.append(tf.reduce_min(inputs, axis=axis))
            self.all_max_values.append(tf.reduce_max(inputs, axis=axis))

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MedianMADStatisticCollector:
    """Use three-sigma approach.
    Constant factor depends on the distribution form. Assuming normal distribution - the factor is 1.4826.
    """
    def __init__(self, per_channel: bool, channel_axes: int):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.samples = []
        self.median = None
        self.mad = None

    @property
    def min(self) -> np.ndarray:
        return (self.median - 3 * 1.726 * self.mad).astype(np.float32)

    @property
    def max(self) -> np.ndarray:
        return (self.median + 3 * 1.726 * self.mad).astype(np.float32)

    def prepare_statistics(self):
        ndims = len(self.samples[0].shape)
        axis = get_axes(ndims, self.per_channel, self.channel_axes, add_dim=True)

        inputs_tensor = np.array(self.samples)
        if self.per_channel:
            per_channel_histories = get_per_channel_history(inputs_tensor, axis)
            per_channel_median = []
            per_channel_mad = []
            for channel_history in per_channel_histories:
                channel_history = discard_zeros(channel_history)
                median = np.median(channel_history)
                per_channel_median.append(median)
                inputs_median_diff = abs(channel_history - median)
                per_channel_mad.append(np.median(inputs_median_diff))
            self.median = np.array(per_channel_median)
            self.mad = np.array(per_channel_mad)
        else:
            inputs_tensor = inputs_tensor.flatten()
            inputs_tensor_flat = discard_zeros(inputs_tensor)
            self.median = np.median(inputs_tensor_flat)
            self.mad = np.median(abs(inputs_tensor_flat - self.median))

    def call(self, inputs: tf.Tensor):
        self.samples.append(inputs.numpy())

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class PercentileStatisticCollector:
    def __init__(self, per_channel: bool, channel_axes: int, min_percentile: float, max_percentile: float):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.samples = []
        self.all_min_values = None
        self.all_max_values = None

    @property
    def min(self) -> np.ndarray:
        return self.all_min_values.astype(np.float32)

    @property
    def max(self) -> np.ndarray:
        return self.all_max_values.astype(np.float32)

    def prepare_statistics(self):
        ndims = len(self.samples[0].shape)
        axis = get_axes(ndims, self.per_channel, self.channel_axes, add_dim=True)

        inputs_tensor = np.array(self.samples)
        if self.per_channel:
            per_channel_histories = get_per_channel_history(inputs_tensor, axis)
            per_channel_max_vals = []
            per_channel_min_vals = []
            for channel_history in per_channel_histories:
                min_val = np.percentile(channel_history, self.min_percentile)
                max_val = np.percentile(channel_history, self.max_percentile)
                per_channel_min_vals.append(min_val)
                per_channel_max_vals.append(max_val)
            self.all_min_values = np.array(per_channel_min_vals)
            self.all_max_values = np.array(per_channel_max_vals)
        else:
            inputs_tensor_flat = inputs_tensor.flatten()
            self.all_min_values = np.percentile(inputs_tensor_flat, self.min_percentile)
            self.all_max_values = np.percentile(inputs_tensor_flat, self.max_percentile)

    def call(self, inputs: tf.Tensor):
        self.samples.append(inputs.numpy())

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MeanPercentileStatisticCollector:
    def __init__(self, per_channel: bool, channel_axes: int, input_type: str,
                 min_percentile: float, max_percentile: float):
        self.per_channel = per_channel
        self.channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.input_type = input_type
        self.all_min_values = []
        self.all_max_values = []

    @property
    def min(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self.all_min_values), axis=0)

    @property
    def max(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self.all_max_values), axis=0)

    def prepare_statistics(self):
        if self.per_channel:
            new_shape = 1
            for val in self.all_min_values[0].shape:
                new_shape *= val
            for i, _ in enumerate(self.all_min_values):
                self.all_min_values[i] = tf.reshape(self.all_min_values[i], shape=(new_shape))
                self.all_max_values[i] = tf.reshape(self.all_max_values[i], shape=(new_shape))

    def _percentile(self, inputs: tf.Tensor, pc: float, axis: list):
        return np.percentile(inputs.numpy(), pc, axis)

    def call(self, inputs: tf.Tensor):
        ndims = len(inputs.shape)
        axis = get_axes(ndims, self.per_channel, self.channel_axes)

        if self.input_type == InputType.INPUTS:
            axis.remove(0)
            min_vals = tf.py_function(self._percentile, [inputs, self.min_percentile, axis], Tout=tf.float32)
            max_vals = tf.py_function(self._percentile, [inputs, self.max_percentile, axis], Tout=tf.float32)
            self.all_min_values.extend(tf.unstack(min_vals))
            self.all_max_values.extend(tf.unstack(max_vals))
        elif self.input_type == InputType.WEIGHTS:
            min_vals = tf.py_function(self._percentile, [inputs, self.min_percentile, axis], Tout=tf.float32)
            max_vals = tf.py_function(self._percentile, [inputs, self.max_percentile, axis], Tout=tf.float32)
            self.all_min_values.append(min_vals)
            self.all_max_values.append(max_vals)

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)
