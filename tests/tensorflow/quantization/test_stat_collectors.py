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

import itertools

import pytest
import numpy as np
import tensorflow as tf

from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.initializers.collectors import MinMaxStatisticsCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanMinMaxStatisticsCollector
from nncf.tensorflow.quantization.initializers.collectors import MedianMADStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import PercentileStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanPercentileStatisticCollector


BATCH_SIZE = 2
HW_SIZE = 5
NUM_CHANNELS = 3
INPUT_TENSOR = tf.range((BATCH_SIZE * HW_SIZE * HW_SIZE * NUM_CHANNELS), dtype=tf.float32)
INPUT_TENSOR = tf.reshape(INPUT_TENSOR, [BATCH_SIZE, HW_SIZE, HW_SIZE, NUM_CHANNELS]) # NHWC: [2, 5, 5, 3]

CHANNEL_AXIS = -1


class TestStatisticCollectorsWithStatAggregation:
    def get_num_samples_per_step(self, input_type):
        if input_type == InputType.INPUTS:
            num_samples_per_step = BATCH_SIZE
        if input_type == InputType.WEIGHTS:
            num_samples_per_step= 1
        return num_samples_per_step

    def check_num_samples(self, collector, num_samples_per_step, step_num):
        assert len(collector.all_min_values) == num_samples_per_step * step_num
        assert len(collector.all_max_values) == num_samples_per_step * step_num

    def check_stats_per_channel_tf_tensor(self, collector,
                                          input_tensor_min_per_channel,
                                          input_tensor_max_per_channel):
        assert tf.math.reduce_all(collector.min.numpy() == pytest.approx(input_tensor_min_per_channel.numpy(), 0.01))
        assert tf.math.reduce_all(collector.max.numpy() == pytest.approx(input_tensor_max_per_channel.numpy(), 0.01))

    def check_stats_per_tensor_tf_tensor(self, collector,
                                         input_tensor_min,
                                         input_tensor_max):
        assert collector.min.numpy() == pytest.approx(input_tensor_min.numpy(), 0.01)
        assert collector.max.numpy() == pytest.approx(input_tensor_max.numpy(), 0.01)

    def check_all_min_max_values_shape(self, per_channel, collector, num_samples_per_step, step_num):
        if per_channel:
            assert tf.stack(collector.all_min_values).shape == (num_samples_per_step * step_num, NUM_CHANNELS)
        if not per_channel:
            assert tf.stack(collector.all_min_values).shape == (num_samples_per_step * step_num,)

    def run_all_checks(self, collector, input_type, per_channel,
                       input_tensor_min_per_channel=None,
                       input_tensor_max_per_channel=None,
                       input_tensor_min=None,
                       input_tensor_max=None):
        # Before first call - no statistics
        assert collector.all_min_values == []
        assert collector.all_max_values == []

        num_samples_per_step = self.get_num_samples_per_step(input_type)
        for step_num in [1, 2]:
            collector(INPUT_TENSOR)
            collector.prepare_statistics()

            self.check_num_samples(collector, num_samples_per_step, step_num)
            if per_channel:
                self.check_stats_per_channel_tf_tensor(collector,
                                                  input_tensor_min_per_channel,
                                                  input_tensor_max_per_channel)
            if not per_channel:
                self.check_stats_per_tensor_tf_tensor(collector,
                                                 input_tensor_min,
                                                 input_tensor_max)
            self.check_all_min_max_values_shape(per_channel, collector, num_samples_per_step, step_num)

    @pytest.mark.parametrize("per_channel, input_type",
                             itertools.product([False, True],
                                               [InputType.INPUTS, InputType.WEIGHTS]))
    def test_min_max(self, per_channel, input_type):
        collector = MinMaxStatisticsCollector(per_channel, CHANNEL_AXIS, input_type)

        if per_channel:
            # Get reference values
            input_tensor_min_per_channel = tf.math.reduce_min(INPUT_TENSOR, axis=(0, 1, 2))
            input_tensor_max_per_channel = tf.math.reduce_max(INPUT_TENSOR, axis=(0, 1, 2))

            self.run_all_checks(collector, input_type, per_channel,
                                input_tensor_min_per_channel=input_tensor_min_per_channel,
                                input_tensor_max_per_channel=input_tensor_max_per_channel)
        if not per_channel:
            # Get reference values
            input_tensor_min = tf.math.reduce_min(INPUT_TENSOR)
            input_tensor_max = tf.math.reduce_max(INPUT_TENSOR)

            self.run_all_checks(collector, input_type, per_channel,
                               input_tensor_min=input_tensor_min,
                               input_tensor_max=input_tensor_max)

    @pytest.mark.parametrize("per_channel, input_type",
                             itertools.product([False, True],
                                               [InputType.INPUTS, InputType.WEIGHTS]))
    def test_mean_min_max(self, per_channel, input_type):
        collector = MeanMinMaxStatisticsCollector(per_channel, CHANNEL_AXIS, input_type)

        axis = [0, 1, 2, 3]
        if per_channel:
            axis.pop()
        if input_type == InputType.INPUTS:
            axis.remove(0)
            reduce_mean_axis = 0
        if input_type == InputType.WEIGHTS:
            reduce_mean_axis = None

        if per_channel:
            # Get reference values
            if input_type == InputType.INPUTS:
                input_tensor_min_per_channel = tf.math.reduce_mean(tf.math.reduce_min(INPUT_TENSOR, axis=axis),
                                                                   axis=reduce_mean_axis)
                input_tensor_max_per_channel = tf.math.reduce_mean(tf.math.reduce_max(INPUT_TENSOR, axis=axis),
                                                                   axis=reduce_mean_axis)
            if input_type == InputType.WEIGHTS:
                input_tensor_min_per_channel = tf.math.reduce_min(INPUT_TENSOR, axis=axis)
                input_tensor_max_per_channel = tf.math.reduce_max(INPUT_TENSOR, axis=axis)

            self.run_all_checks(collector, input_type, per_channel,
                                input_tensor_min_per_channel=input_tensor_min_per_channel,
                                input_tensor_max_per_channel=input_tensor_max_per_channel)
        if not per_channel:
            # Get reference values
            input_tensor_min = tf.math.reduce_mean(tf.math.reduce_min(INPUT_TENSOR, axis=axis),
                                                   axis=reduce_mean_axis)
            input_tensor_max = tf.math.reduce_mean(tf.math.reduce_max(INPUT_TENSOR, axis=axis),
                                                   axis=reduce_mean_axis)

            self.run_all_checks(collector, input_type, per_channel,
                               input_tensor_min=input_tensor_min,
                               input_tensor_max=input_tensor_max)

    @pytest.mark.parametrize("per_channel, input_type, percentiles",
                             itertools.product([True, False],
                                               [InputType.INPUTS, InputType.WEIGHTS],
                                               [[0.0, 100.0], [0.1, 99.9], [33.3, 66.6]]))
    def test_mean_percentile(self, per_channel, input_type, percentiles):
        min_percentile = percentiles[0]
        max_percentile = percentiles[1]
        collector = MeanPercentileStatisticCollector(per_channel, CHANNEL_AXIS, input_type,
                                                     min_percentile, max_percentile)

        axis = [0, 1, 2, 3]
        if per_channel:
            axis.pop()
        if input_type == InputType.INPUTS:
            axis.remove(0)
            reduce_mean_axis = 0
        if input_type == InputType.WEIGHTS:
            reduce_mean_axis = None

        def _percentile(inputs: tf.Tensor, pc: int, axis: list):
            return np.percentile(inputs.numpy(), pc, axis)

        if per_channel:
            if input_type == InputType.INPUTS:
                min_vals = _percentile(INPUT_TENSOR, min_percentile, axis)
                input_tensor_min_per_channel = tf.math.reduce_mean(min_vals, axis=reduce_mean_axis)
                max_vals = _percentile(INPUT_TENSOR, max_percentile, axis)
                input_tensor_max_per_channel = tf.math.reduce_mean(max_vals, axis=reduce_mean_axis)
            if input_type == InputType.WEIGHTS:
                input_tensor_min_per_channel = _percentile(INPUT_TENSOR, min_percentile, axis)
                input_tensor_max_per_channel = _percentile(INPUT_TENSOR, max_percentile, axis)
            input_tensor_min_per_channel = tf.cast(input_tensor_min_per_channel, tf.float32)
            input_tensor_max_per_channel = tf.cast(input_tensor_max_per_channel, tf.float32)
            self.run_all_checks(collector, input_type, per_channel,
                                input_tensor_min_per_channel=input_tensor_min_per_channel,
                                input_tensor_max_per_channel=input_tensor_max_per_channel)
        if not per_channel:
            min_vals = _percentile(INPUT_TENSOR, min_percentile, axis)
            input_tensor_min = tf.math.reduce_mean(tf.cast(min_vals, tf.float32), axis=reduce_mean_axis)
            max_vals = _percentile(INPUT_TENSOR, max_percentile, axis)
            input_tensor_max = tf.math.reduce_mean(tf.cast(max_vals, tf.float32), axis=reduce_mean_axis)
            self.run_all_checks(collector, input_type, per_channel,
                               input_tensor_min=input_tensor_min,
                               input_tensor_max=input_tensor_max)


class TestStatisticCollectorsWithDataAggregation:
    def check_stats_per_channel_np(self, collector,
                                   input_tensor_min_per_channel,
                                   input_tensor_max_per_channel):
        assert tf.math.reduce_all(collector.min == pytest.approx(input_tensor_min_per_channel, 0.01))
        assert tf.math.reduce_all(collector.max == pytest.approx(input_tensor_max_per_channel, 0.01))

    def check_stats_per_tensor_np(self, collector,
                                  input_tensor_min,
                                  input_tensor_max):
        assert collector.min == pytest.approx(input_tensor_min, 0.01)
        assert collector.max == pytest.approx(input_tensor_max, 0.01)

    def run_all_checks(self, collector, step_num, per_channel,
                       input_tensor_min_per_channel=None,
                       input_tensor_max_per_channel=None,
                       input_tensor_min=None,
                       input_tensor_max=None):
        assert len(collector.samples) == step_num

        if per_channel:
            self.check_stats_per_channel_np(collector,
                                            input_tensor_min_per_channel,
                                            input_tensor_max_per_channel)
        if not per_channel:
            self.check_stats_per_tensor_np(collector,
                                           input_tensor_min,
                                           input_tensor_max)

    @pytest.mark.parametrize("per_channel, input_type, percentiles",
                             itertools.product([True, False],
                                               [InputType.INPUTS, InputType.WEIGHTS],
                                               [[0.0, 100.0], [0.1, 99.9], [33.3, 66.6]]))
    def test_percentile(self, per_channel, input_type, percentiles):
        min_percentile = percentiles[0]
        max_percentile = percentiles[1]
        collector = PercentileStatisticCollector(per_channel, CHANNEL_AXIS, min_percentile, max_percentile)

        axis = [0, 1, 2, 3]
        # all input tensors are stacked together - one more dimension
        axis.append(axis[-1] + 1)
        if per_channel:
            axis.pop()

        # Before first call - no samples
        assert collector.samples == []

        for step_num in [1, 2]:
            collector(INPUT_TENSOR)
            collector.prepare_statistics()

            # Get reference values
            samples = np.array([INPUT_TENSOR for _ in range(step_num)])
            if per_channel:
                input_tensor_min_per_channel = np.percentile(samples, min_percentile, axis)
                input_tensor_max_per_channel = np.percentile(samples, max_percentile, axis)
                input_tensor_min_per_channel = tf.cast(input_tensor_min_per_channel, tf.float32)
                input_tensor_max_per_channel = tf.cast(input_tensor_max_per_channel, tf.float32)
            if not per_channel:
                input_tensor_min = np.percentile(samples, min_percentile)
                input_tensor_max = np.percentile(samples, max_percentile)

            if per_channel:
                self.run_all_checks(collector, step_num, per_channel,
                                    input_tensor_min_per_channel=input_tensor_min_per_channel,
                                    input_tensor_max_per_channel=input_tensor_max_per_channel)
            if not per_channel:
                self.run_all_checks(collector, step_num, per_channel,
                                    input_tensor_min=input_tensor_min,
                                    input_tensor_max=input_tensor_max)

    @pytest.mark.parametrize("per_channel, input_type",
                             itertools.product([False, True],
                                               [InputType.INPUTS, InputType.WEIGHTS]))
    def test_threesigma(self, per_channel, input_type):
        collector = MedianMADStatisticCollector(per_channel, CHANNEL_AXIS)

        input_tensor = tf.range(1, (BATCH_SIZE * HW_SIZE * HW_SIZE * NUM_CHANNELS + 1), dtype=tf.float32)
        input_tensor = tf.reshape(input_tensor, [BATCH_SIZE, HW_SIZE, HW_SIZE, NUM_CHANNELS])

        axis = [0, 1, 2, 3]
        # all input tensors are stacked together - one more dimension
        axis.append(axis[-1] + 1)
        if per_channel:
            axis.pop()

        # Before first call - no samples
        assert collector.samples == []

        for step_num in [1, 2]:
            collector(input_tensor)
            collector.prepare_statistics()

            # Get reference values
            samples = np.array([input_tensor for _ in range(step_num)])
            if per_channel:
                median = np.median(samples, axis=axis)
                mad = np.median(abs(samples - median))
                input_tensor_min_per_channel = (median - 3 * 1.726 * mad).astype(np.float32)
                input_tensor_max_per_channel = (median + 3 * 1.726 * mad).astype(np.float32)
            if not per_channel:
                inputs_tensor_flat = samples.flatten()
                median = np.median(inputs_tensor_flat)
                mad = np.median(abs(inputs_tensor_flat - median))
                input_tensor_min = (median - 3 * 1.726 * mad).astype(np.float32)
                input_tensor_max = (median + 3 * 1.726 * mad).astype(np.float32)

            if per_channel:
                self.run_all_checks(collector, step_num, per_channel,
                                    input_tensor_min_per_channel=input_tensor_min_per_channel,
                                    input_tensor_max_per_channel=input_tensor_max_per_channel)
            if not per_channel:
                self.run_all_checks(collector, step_num, per_channel,
                                    input_tensor_min=input_tensor_min,
                                    input_tensor_max=input_tensor_max)
