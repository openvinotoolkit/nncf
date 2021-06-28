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

from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf

from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.initializers.utils import get_per_channel_history
from nncf.tensorflow.quantization.initializers.utils import discard_zeros
from nncf.tensorflow.quantization.initializers.utils import get_axes


class MinMaxStatisticCollector:
    """
    Collector uses min of minimum values and max of maximum values.
    """

    def __init__(self, per_channel: bool, channel_axes: Union[int, Tuple[int], List[int]], input_type: str):
        self._per_channel = per_channel
        self._channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self._input_type = input_type
        self._all_min_values = []
        self._all_max_values = []

    @property
    def min(self) -> tf.Tensor:
        return tf.math.reduce_min(tf.stack(self._all_min_values), axis=0)

    @property
    def max(self) -> tf.Tensor:
        return tf.math.reduce_max(tf.stack(self._all_max_values), axis=0)

    def prepare_statistics(self):
        if self._per_channel:
            new_shape = np.prod(self._all_min_values[0].shape).item()
            for i, _ in enumerate(self._all_min_values):
                self._all_min_values[i] = tf.reshape(self._all_min_values[i], shape=new_shape)
                self._all_max_values[i] = tf.reshape(self._all_max_values[i], shape=new_shape)

    def call(self, inputs: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._all_min_values or self._input_type == InputType.INPUTS:
            ndims = len(inputs.shape)
            axis = get_axes(ndims, self._per_channel, self._channel_axes)

            if self._input_type == InputType.INPUTS:
                axis.remove(0)
                self._all_min_values.extend(tf.unstack(tf.reduce_min(inputs, axis=axis)))
                self._all_max_values.extend(tf.unstack(tf.reduce_max(inputs, axis=axis)))
            elif self._input_type == InputType.WEIGHTS:
                self._all_min_values.append(tf.reduce_min(inputs, axis=axis))
                self._all_max_values.append(tf.reduce_max(inputs, axis=axis))

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MeanMinMaxStatisticsCollector:
    """
    Collector uses mean of minimum values and mean of maximum values.
    """

    def __init__(self, per_channel: bool, channel_axes: Union[int, Tuple[int], List[int]], input_type: str):
        self._per_channel = per_channel
        self._channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self._input_type = input_type
        self._all_min_values = []
        self._all_max_values = []

    @property
    def min(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self._all_min_values), axis=0)

    @property
    def max(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self._all_max_values), axis=0)

    def prepare_statistics(self):
        if self._per_channel:
            new_shape = np.prod(self._all_min_values[0].shape).item()
            for i, _ in enumerate(self._all_min_values):
                self._all_min_values[i] = tf.reshape(self._all_min_values[i], shape=new_shape)
                self._all_max_values[i] = tf.reshape(self._all_max_values[i], shape=new_shape)

    def call(self, inputs: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._all_min_values or self._input_type == InputType.INPUTS:
            ndims = len(inputs.shape)
            axis = get_axes(ndims, self._per_channel, self._channel_axes)

            if self._input_type == InputType.INPUTS:
                axis.remove(0)
                self._all_min_values.extend(tf.unstack(tf.reduce_min(inputs, axis=axis)))
                self._all_max_values.extend(tf.unstack(tf.reduce_max(inputs, axis=axis)))
            elif self._input_type == InputType.WEIGHTS:
                self._all_min_values.append(tf.reduce_min(inputs, axis=axis))
                self._all_max_values.append(tf.reduce_max(inputs, axis=axis))

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MedianMADStatisticCollector:
    """
    Collector uses three-sigma approach with the assumption of normal distribution by default.
    """

    def __init__(self, per_channel: bool, channel_axes: Union[int, Tuple[int], List[int]], input_type: str):
        self._per_channel = per_channel
        self._channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self._input_type = input_type
        self._samples = []
        self._median = None
        self._mad = None

        # Constant factor depends on the distribution form. Assuming normal distribution - the factor is 1.4826.
        self.distribution_factor = 1.4826230

    @property
    def min(self) -> np.ndarray:
        return (self._median - 3 * self.distribution_factor * self._mad).astype(np.float32)

    @property
    def max(self) -> np.ndarray:
        return (self._median + 3 * self.distribution_factor * self._mad).astype(np.float32)

    def prepare_statistics(self):
        ndims = len(self._samples[0].shape)
        axis = get_axes(ndims, self._per_channel, self._channel_axes, add_dim=True)

        inputs_tensor = np.array(self._samples)
        if self._per_channel:
            per_channel_histories = get_per_channel_history(inputs_tensor, axis)
            per_channel_median = []
            per_channel_mad = []
            for channel_history in per_channel_histories:
                channel_history = discard_zeros(channel_history)
                median = np.median(channel_history)
                per_channel_median.append(median)
                inputs_median_diff = abs(channel_history - median)
                per_channel_mad.append(np.median(inputs_median_diff))
            self._median = np.array(per_channel_median)
            self._mad = np.array(per_channel_mad)
        else:
            inputs_tensor = inputs_tensor.flatten()
            inputs_tensor_flat = discard_zeros(inputs_tensor)
            self._median = np.median(inputs_tensor_flat)
            self._mad = np.median(abs(inputs_tensor_flat - self._median))

    def call(self, inputs: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._samples or self._input_type == InputType.INPUTS:
            self._samples.append(inputs.numpy())

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class PercentileStatisticCollector:
    """
    Collector uses percentiles to estimate min and max of all data history.
    """

    def __init__(self, per_channel: bool, channel_axes: Union[int, Tuple[int], List[int]], input_type: str,
                 min_percentile: float, max_percentile: float):
        self._per_channel = per_channel
        self._channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self._input_type = input_type
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile
        self._samples = []
        self._min_values = None
        self._max_values = None

    @property
    def min(self) -> np.ndarray:
        return self._min_values.astype(np.float32)

    @property
    def max(self) -> np.ndarray:
        return self._max_values.astype(np.float32)

    def prepare_statistics(self):
        ndims = len(self._samples[0].shape)
        axis = get_axes(ndims, self._per_channel, self._channel_axes, add_dim=True)

        inputs_tensor = np.array(self._samples)
        if self._per_channel:
            per_channel_histories = get_per_channel_history(inputs_tensor, axis)
            per_channel_max_vals = []
            per_channel_min_vals = []
            for channel_history in per_channel_histories:
                min_val = np.percentile(channel_history, self._min_percentile)
                max_val = np.percentile(channel_history, self._max_percentile)
                per_channel_min_vals.append(min_val)
                per_channel_max_vals.append(max_val)
            self._min_values = np.array(per_channel_min_vals)
            self._max_values = np.array(per_channel_max_vals)
        else:
            inputs_tensor_flat = inputs_tensor.flatten()
            self._min_values = np.percentile(inputs_tensor_flat, self._min_percentile)
            self._max_values = np.percentile(inputs_tensor_flat, self._max_percentile)

    def call(self, inputs: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._samples or self._input_type == InputType.INPUTS:
            self._samples.append(inputs.numpy())

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class MeanPercentileStatisticCollector:
    """
    Collector uses percentiles to estimate min and max of data per step
    and then averages the statistics.
    """

    def __init__(self, per_channel: bool, channel_axes: Union[int, Tuple[int], List[int]], input_type: str,
                 min_percentile: float, max_percentile: float):
        self._per_channel = per_channel
        self._channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
        self._input_type = input_type
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile
        self._all_min_values = []
        self._all_max_values = []

    @property
    def min(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self._all_min_values), axis=0)

    @property
    def max(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self._all_max_values), axis=0)

    def prepare_statistics(self):
        if self._per_channel:
            new_shape = np.prod(self._all_min_values[0].shape).item()
            for i, _ in enumerate(self._all_min_values):
                self._all_min_values[i] = tf.reshape(self._all_min_values[i], shape=new_shape)
                self._all_max_values[i] = tf.reshape(self._all_max_values[i], shape=new_shape)

    def _percentile(self, inputs: tf.Tensor, pc: float, axis: list):
        return np.percentile(inputs.numpy(), pc, axis)

    def call(self, inputs: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._all_min_values or self._input_type == InputType.INPUTS:
            ndims = len(inputs.shape)
            axis = get_axes(ndims, self._per_channel, self._channel_axes)

            if self._input_type == InputType.INPUTS:
                axis.remove(0)
                min_vals = tf.py_function(self._percentile, [inputs, self._min_percentile, axis], Tout=tf.float32)
                max_vals = tf.py_function(self._percentile, [inputs, self._max_percentile, axis], Tout=tf.float32)
                self._all_min_values.extend(tf.unstack(min_vals))
                self._all_max_values.extend(tf.unstack(max_vals))
            elif self._input_type == InputType.WEIGHTS:
                min_vals = tf.py_function(self._percentile, [inputs, self._min_percentile, axis], Tout=tf.float32)
                max_vals = tf.py_function(self._percentile, [inputs, self._max_percentile, axis], Tout=tf.float32)
                self._all_min_values.append(min_vals)
                self._all_max_values.append(max_vals)

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)
