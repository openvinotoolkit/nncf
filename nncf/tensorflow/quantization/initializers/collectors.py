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

from collections import deque
from typing import List, Union

import numpy as np
import tensorflow as tf

from nncf.common.tensor_statistics.collectors import OfflineTensorStatisticCollector
from nncf.common.tensor_statistics.collectors import MixedMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import Aggregator
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.tensorflow.quantization.initializers.utils import get_per_channel_history
from nncf.tensorflow.quantization.initializers.utils import discard_zeros


class TFAggregator(Aggregator):
    @staticmethod
    def reduce_min(x: tf.Tensor, reduction_shape: Union[int, tuple, list]) -> tf.Tensor:
        return tf.squeeze(tf.reduce_min(x, axis=reduction_shape))

    @staticmethod
    def reduce_max(x: tf.Tensor, reduction_shape: Union[int, tuple, list]) -> tf.Tensor:
        return tf.squeeze(tf.reduce_max(x, axis=reduction_shape))

    @staticmethod
    def abs(x: tf.Tensor) -> tf.Tensor:
        return tf.math.abs(x)

    @staticmethod
    def min(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        return tf.math.minimum(x1, x2)

    @staticmethod
    def max(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        return tf.math.maximum(x1, x2)

    @classmethod
    def tensor_min(cls, x: tf.Tensor, axis: Union[int, tuple, list]) -> tf.Tensor:
        return cls.reduce_min(x, axis)

    @classmethod
    def tensor_max(cls, x: tf.Tensor, axis: Union[int, tuple, list]) -> tf.Tensor:
        return cls.reduce_max(x, axis)

    @staticmethod
    def mean(x: tf.Tensor, axis: Union[int, tuple, list]) -> tf.Tensor:
        return tf.math.reduce_mean(x, axis=axis)

    @staticmethod
    def stack(x: deque) -> tf.Tensor:
        return tf.stack(x)

    @staticmethod
    def list_to_extend_stat_history(x: tf.Tensor) -> tf.Tensor:
        return tf.unstack(x)


class TFMinMaxStatisticCollector(MinMaxStatisticCollector):
    @staticmethod
    def _get_aggregator() -> Aggregator:
        return TFAggregator()

    def _register_input(self, x: tf.Tensor):
        self._register_input_common(x)


class TFMixedMinMaxStatisticCollector(MixedMinMaxStatisticCollector):
    @staticmethod
    def _get_aggregator() -> Aggregator:
        return TFAggregator()

    def _register_input(self, x: tf.Tensor):
        self._register_input_common(x)


class TFMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    @staticmethod
    def _get_aggregator() -> Aggregator:
        return TFAggregator()

    def _register_input(self, x: tf.Tensor):
        self._register_input_common(x)


class MedianMADStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates median and median absolute deviation (MAD).
    """

    def _register_input(self, x: tf.Tensor):
        self._samples.append(x.numpy())

    def _get_statistics(self) -> MedianMADTensorStatistic:
        # all input tensors are stacked together - one more dimension
        self._reduction_shape = self._reduction_shape + (self._reduction_shape[-1] + 1,)

        per_channel_histories = get_per_channel_history(np.array(self._samples), list(self._reduction_shape))
        per_channel_median = []
        per_channel_mad = []
        for channel_history in per_channel_histories:
            channel_history = discard_zeros(channel_history)
            median = np.median(channel_history)
            per_channel_median.append(median)
            inputs_median_diff = abs(channel_history - median)
            per_channel_mad.append(np.median(inputs_median_diff))
        median = tf.squeeze(tf.convert_to_tensor(np.array(per_channel_median), dtype=tf.float32))
        mad = tf.squeeze(tf.convert_to_tensor(np.array(per_channel_mad), dtype=tf.float32))
        return MedianMADTensorStatistic(median, mad)


class PercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values of all data history.
    """

    def __init__(self,
                 reduction_shape: ReductionShape,
                 percentiles_to_collect: List[float],
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect

    def _register_input(self, x: tf.Tensor):
        self._samples.append(x.numpy())

    def _get_statistics(self) -> PercentileTensorStatistic:
        # all input tensors are stacked together - one more dimension
        self._reduction_shape = self._reduction_shape + (self._reduction_shape[-1] + 1,)

        percentile_vs_values_dict = {}
        for pc in self._percentiles_to_collect:
            per_channel_histories = get_per_channel_history(np.array(self._samples), list(self._reduction_shape))
            per_channel_vals = []
            for channel_history in per_channel_histories:
                channel_history = discard_zeros(channel_history)
                val = np.percentile(channel_history, pc)
                per_channel_vals.append(val)
            percentile_vs_values_dict[pc] = tf.squeeze(tf.convert_to_tensor(np.array(per_channel_vals),
                                                                            dtype=tf.float32))
        return PercentileTensorStatistic(percentile_vs_values_dict)


class MeanPercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values per step and then averages the results.
    """

    def __init__(self,
                 reduction_shape: ReductionShape,
                 percentiles_to_collect: List[float],
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._use_per_sample_stats = 0 not in reduction_shape # assume batch is the first dimension
        self._all_pct_values = {}
        for pc in percentiles_to_collect:
            self._all_pct_values[pc] = deque()

    def _percentile(self, inputs: tf.Tensor, pc: float, axis: list) -> np.ndarray:
        return np.percentile(inputs.numpy(), pc, axis)

    def _register_input(self, x: tf.Tensor):
        for pct, values in self._all_pct_values.items():
            if self._use_per_sample_stats:
                vals = tf.squeeze(tf.py_function(self._percentile, [x, pct, self._reduction_shape], Tout=tf.float32))
                values.extend(tf.unstack(vals))
            else:
                vals = tf.squeeze(tf.py_function(self._percentile, [x, pct, self._reduction_shape], Tout=tf.float32))
                values.append(vals)

    def _get_statistics(self) -> PercentileTensorStatistic:
        mean_percentile_values = {}
        for pct, values in self._all_pct_values.items():
            stacked_pct_vals = tf.stack(values)
            mean_percentile_values[pct] = tf.math.reduce_mean(stacked_pct_vals, axis=0)
        return PercentileTensorStatistic(mean_percentile_values)
