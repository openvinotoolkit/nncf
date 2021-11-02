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

from abc import ABC
from collections import deque
from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf

from nncf.common.tensor_statistics.collectors import OfflineTensorStatisticCollector
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.initializers.utils import get_per_channel_history
from nncf.tensorflow.quantization.initializers.utils import discard_zeros
from nncf.tensorflow.quantization.initializers.utils import get_axes


class TFOfflineTensorStatisticCollector(OfflineTensorStatisticCollector, ABC):
    def __init__(self, input_type: str, channel_axes: Union[int, Tuple[int], List[int]],
                 per_channel: bool = False, num_samples: int = None):
        super().__init__(num_samples=num_samples)
        self._per_channel = per_channel
        self._input_type = input_type
        self._channel_axes = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]

    def _register_input(self, x: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._samples or self._input_type == InputType.INPUTS:
            self._samples.append(x.numpy())

    def __call__(self, *args, **kwargs):
        self.register_input(*args, **kwargs)


class MinMaxStatisticCollectorBase(TFOfflineTensorStatisticCollector, ABC):
    def __init__(self, channel_axes: Union[int, Tuple[int], List[int]], input_type: str,
                 mode: str = 'symmetric', per_channel: bool = False, num_samples: int = None, window_size: int = None):
        super().__init__(input_type, channel_axes, per_channel, num_samples)
        self._all_min_values = deque(maxlen=window_size)
        self._all_max_values = deque(maxlen=window_size)
        self._mode = mode

    def _register_input(self, x: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not self._all_min_values or self._input_type == InputType.INPUTS:
            ndims = len(x.shape)
            axis = get_axes(ndims, self._per_channel, self._channel_axes)

            if self._input_type == InputType.INPUTS:
                axis.remove(0)
            min_reduced = tf.reduce_min(x, axis=axis)
            if self._mode == 'symmetric':
                inputs = tf.math.abs(x)
            max_reduced = tf.reduce_max(inputs, axis=axis)

            if self._input_type == InputType.INPUTS:
                self._all_min_values.extend(tf.unstack(min_reduced))
                self._all_max_values.extend(tf.unstack(max_reduced))
            elif self._input_type == InputType.WEIGHTS:
                self._all_min_values.append(min_reduced)
                self._all_max_values.append(max_reduced)

    def _prepare_statistics(self):
        if self._per_channel:
            new_shape = np.prod(self._all_min_values[0].shape).item()
            for i, _ in enumerate(self._all_min_values):
                self._all_min_values[i] = tf.reshape(self._all_min_values[i], shape=new_shape)
                self._all_max_values[i] = tf.reshape(self._all_max_values[i], shape=new_shape)


class MinMaxStatisticCollector(MinMaxStatisticCollectorBase):
    """
    Collector aggregates min of minimum values and max of maximum values.
    """

    def _min_aggregate(self) -> tf.Tensor:
        return tf.math.reduce_min(tf.stack(self._all_min_values), axis=0)

    def _max_aggregate(self) -> tf.Tensor:
        return tf.math.reduce_max(tf.stack(self._all_max_values), axis=0)

    def _get_statistics(self) -> MinMaxTensorStatistic:
        self._prepare_statistics()
        return MinMaxTensorStatistic(self._min_aggregate(), self._max_aggregate())


class MixedMinMaxStatisticCollector(MinMaxStatisticCollectorBase):
    """
    Collector aggregates (min or mean) of minimum values and (max or mean) of maximum values.
    """

    def _min_aggregate(self) -> tf.Tensor:
        if self._input_type == InputType.INPUTS and not self._per_channel and self._mode == 'asymmetric':
            return tf.math.reduce_mean(tf.stack(self._all_min_values), axis=0)
        return tf.math.reduce_min(tf.stack(self._all_min_values), axis=0)

    def _max_aggregate(self) -> tf.Tensor:
        if self._input_type == InputType.INPUTS and not self._per_channel:
            return tf.math.reduce_mean(tf.stack(self._all_max_values), axis=0)
        return tf.math.reduce_max(tf.stack(self._all_max_values), axis=0)

    def _get_statistics(self) -> MinMaxTensorStatistic:
        self._prepare_statistics()
        return MinMaxTensorStatistic(self._min_aggregate(), self._max_aggregate())


class MeanMinMaxStatisticsCollector(MinMaxStatisticCollectorBase):
    """
    Collector aggregates mean of minimum values and mean of maximum values.
    """

    def _min_aggregate(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self._all_min_values), axis=0)

    def _max_aggregate(self) -> tf.Tensor:
        return tf.math.reduce_mean(tf.stack(self._all_max_values), axis=0)

    def _get_statistics(self) -> MinMaxTensorStatistic:
        self._prepare_statistics()
        return MinMaxTensorStatistic(self._min_aggregate(), self._max_aggregate())


class MedianMADStatisticCollector(TFOfflineTensorStatisticCollector):
    """
    Collector uses three-sigma approach.
    """

    def __init__(self, channel_axes: Union[int, Tuple[int], List[int]],
                 input_type: str, per_channel: bool = False, num_samples: int = None):
        super().__init__(input_type, channel_axes, per_channel, num_samples)
        self._median = None
        self._mad = None

    def _prepare_statistics(self):
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
            self._median = tf.convert_to_tensor(np.array(per_channel_median), dtype=tf.float32)
            self._mad = tf.convert_to_tensor(np.array(per_channel_mad), dtype=tf.float32)
        else:
            inputs_tensor = inputs_tensor.flatten()
            inputs_tensor_flat = discard_zeros(inputs_tensor)
            self._median = tf.convert_to_tensor(np.median(inputs_tensor_flat), dtype=tf.float32)
            self._mad = tf.convert_to_tensor(np.median(abs(inputs_tensor_flat - self._median)), dtype=tf.float32)

    def _get_statistics(self) -> MedianMADTensorStatistic:
        self._prepare_statistics()
        return MedianMADTensorStatistic(self._median, self._mad)


class PercentileStatisticCollector(TFOfflineTensorStatisticCollector):
    """
    Collector uses percentiles to estimate min and max of all data history.
    """

    def __init__(self, channel_axes: Union[int, Tuple[int], List[int]], input_type: str,
                 percentiles_to_collect: List[float], per_channel: bool = False, num_samples: int = None):
        super().__init__(input_type, channel_axes, per_channel, num_samples)
        self._percentiles_to_collect = percentiles_to_collect
        self.percentile_vs_values_dict = {}

    def _prepare_statistics(self):
        ndims = len(self._samples[0].shape)
        axis = get_axes(ndims, self._per_channel, self._channel_axes, add_dim=True)
        inputs_tensor = np.array(self._samples)
        for pc in self._percentiles_to_collect:
            if self._per_channel:
                per_channel_histories = get_per_channel_history(inputs_tensor, axis)
                per_channel_vals = []
                for channel_history in per_channel_histories:
                    val = np.percentile(channel_history, pc)
                    per_channel_vals.append(val)
                self.percentile_vs_values_dict[pc] = tf.convert_to_tensor(np.array(per_channel_vals), dtype=tf.float32)
            else:
                inputs_tensor_flat = inputs_tensor.flatten()
                self.percentile_vs_values_dict[pc] = tf.convert_to_tensor(np.percentile(inputs_tensor_flat, pc),
                                                                     dtype=tf.float32)

    def _get_statistics(self) -> PercentileTensorStatistic:
        self._prepare_statistics()
        return PercentileTensorStatistic(self.percentile_vs_values_dict)


class MeanPercentileStatisticCollector(TFOfflineTensorStatisticCollector):
    """
    Collector uses percentiles to estimate min and max of data per step
    and then averages the statistics.
    """

    def __init__(self, channel_axes: Union[int, Tuple[int], List[int]], input_type: str,
                 percentiles_to_collect: List[float], per_channel: bool = False, num_samples: int = None):
        super().__init__(input_type, channel_axes, per_channel, num_samples)
        self._all_pct_values = {}
        for pc in percentiles_to_collect:
            self._all_pct_values[pc] = deque()

    def _percentile(self, inputs: tf.Tensor, pc: float, axis: list):
        return np.percentile(inputs.numpy(), pc, axis)

    def _register_input(self, x: tf.Tensor):
        # No need to store extra statistics in memory since weights won't change during range init
        if not list(self._all_pct_values.values())[0] or self._input_type == InputType.INPUTS:
            ndims = len(x.shape)
            axis = get_axes(ndims, self._per_channel, self._channel_axes)

            if self._input_type == InputType.INPUTS:
                axis.remove(0)
            for pct, values in self._all_pct_values.items():
                if self._input_type == InputType.INPUTS:
                    vals = tf.py_function(self._percentile, [x, pct, axis], Tout=tf.float32)
                    values.extend(tf.unstack(vals))
                elif self._input_type == InputType.WEIGHTS:
                    vals = tf.py_function(self._percentile, [x, pct, axis], Tout=tf.float32)
                    values.append(vals)

    def _prepare_statistics(self):
        if self._per_channel:
            new_shape = np.prod(list(self._all_pct_values.values())[0][0].shape).item()
            for values in self._all_pct_values.values():
                for i, _ in enumerate(values):
                    values[i] = tf.reshape(values[i], shape=new_shape)

    def _get_statistics(self) -> PercentileTensorStatistic:
        self._prepare_statistics()
        mean_percentile_values = {}
        for pct, values in self._all_pct_values.items():
            stacked_pct_vals = tf.stack(values)
            mean_percentile_values[pct] = tf.math.reduce_mean(stacked_pct_vals, axis=0)
        return PercentileTensorStatistic(mean_percentile_values)
