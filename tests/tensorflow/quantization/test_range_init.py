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
from collections import namedtuple

import pytest
import numpy as np
import tensorflow as tf

from nncf.common.quantization.initialization.range import PerLayerRangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.structs import QuantizerConfig, QuantizationMode
from nncf.tensorflow.quantization.initializers.collectors import MinMaxStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanMinMaxStatisticsCollector
from nncf.tensorflow.quantization.initializers.collectors import MedianMADStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import PercentileStatisticCollector
from nncf.tensorflow.quantization.initializers.collectors import MeanPercentileStatisticCollector
from nncf.tensorflow.quantization import FakeQuantize
from nncf.tensorflow.quantization.initializers.init_range import TFRangeInitParams
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.wrapper import NNCFWrapper


BATCH_SIZE = 2
HW_SIZE = 5
NUM_CHANNELS = 3
INPUT_TENSOR = tf.range((BATCH_SIZE * HW_SIZE * HW_SIZE * NUM_CHANNELS), dtype=tf.float32)
INPUT_TENSOR = tf.reshape(INPUT_TENSOR, [BATCH_SIZE, HW_SIZE, HW_SIZE, NUM_CHANNELS]) # NHWC: [2, 5, 5, 3]

CHANNEL_AXIS = -1


class TestStatisticCollectorsWithStatAggregation:
    def check_num_samples(self, collector, num_samples_per_step, step_num, input_type):
        if input_type == InputType.INPUTS:
            assert len(collector.all_min_values) == num_samples_per_step * step_num
            assert len(collector.all_max_values) == num_samples_per_step * step_num
        if input_type == InputType.WEIGHTS:
            assert len(collector.all_min_values) == 1
            assert len(collector.all_max_values) == 1

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

    def check_all_min_max_values_shape(self, per_channel, collector, num_samples_per_step_activation,
                                       step_num, input_type):
        if input_type == InputType.INPUTS:
            if per_channel:
                assert tf.stack(collector.all_min_values).shape == (num_samples_per_step_activation * step_num,
                                                                    NUM_CHANNELS)
            if not per_channel:
                assert tf.stack(collector.all_min_values).shape == (num_samples_per_step_activation * step_num,)
        if input_type == InputType.WEIGHTS:
            if per_channel:
                assert tf.stack(collector.all_min_values).shape == (1, NUM_CHANNELS)
            if not per_channel:
                assert tf.stack(collector.all_min_values).shape == (1,)

    def run_all_checks(self, collector, input_type, per_channel,
                       input_tensor_min_per_channel=None,
                       input_tensor_max_per_channel=None,
                       input_tensor_min=None,
                       input_tensor_max=None):
        # Before first call - no statistics
        assert collector.all_min_values == []
        assert collector.all_max_values == []

        num_samples_per_step_activation = BATCH_SIZE
        for step_num in [1, 2]:
            collector(INPUT_TENSOR)
            collector.prepare_statistics()

            self.check_num_samples(collector, num_samples_per_step_activation, step_num, input_type)
            if per_channel:
                self.check_stats_per_channel_tf_tensor(collector,
                                                  input_tensor_min_per_channel,
                                                  input_tensor_max_per_channel)
            if not per_channel:
                self.check_stats_per_tensor_tf_tensor(collector,
                                                 input_tensor_min,
                                                 input_tensor_max)
            self.check_all_min_max_values_shape(per_channel, collector, num_samples_per_step_activation,
                                                step_num, input_type)

    @pytest.mark.parametrize("per_channel, input_type",
                             itertools.product([False, True],
                                               [InputType.INPUTS, InputType.WEIGHTS]))
    def test_min_max(self, per_channel, input_type):
        collector = MinMaxStatisticCollector(per_channel, CHANNEL_AXIS, input_type)

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

    def run_all_checks(self, collector, step_num, per_channel, input_type,
                       input_tensor_min_per_channel=None,
                       input_tensor_max_per_channel=None,
                       input_tensor_min=None,
                       input_tensor_max=None):
        if input_type == InputType.INPUTS:
            assert len(collector.samples) == step_num
        if input_type == InputType.WEIGHTS:
            assert len(collector.samples) == 1

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
        collector = PercentileStatisticCollector(per_channel, CHANNEL_AXIS, input_type, min_percentile, max_percentile)

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
            if input_type == InputType.INPUTS:
                samples = np.array([INPUT_TENSOR for _ in range(step_num)])
            if input_type == InputType.WEIGHTS:
                samples = np.array([INPUT_TENSOR])

            if per_channel:
                input_tensor_min_per_channel = np.percentile(samples, min_percentile, axis)
                input_tensor_max_per_channel = np.percentile(samples, max_percentile, axis)
                input_tensor_min_per_channel = tf.cast(input_tensor_min_per_channel, tf.float32)
                input_tensor_max_per_channel = tf.cast(input_tensor_max_per_channel, tf.float32)
            if not per_channel:
                input_tensor_min = np.percentile(samples, min_percentile)
                input_tensor_max = np.percentile(samples, max_percentile)

            if per_channel:
                self.run_all_checks(collector, step_num, per_channel, input_type,
                                    input_tensor_min_per_channel=input_tensor_min_per_channel,
                                    input_tensor_max_per_channel=input_tensor_max_per_channel)
            if not per_channel:
                self.run_all_checks(collector, step_num, per_channel, input_type,
                                    input_tensor_min=input_tensor_min,
                                    input_tensor_max=input_tensor_max)

    @pytest.mark.parametrize("per_channel, input_type",
                             itertools.product([False, True],
                                               [InputType.INPUTS, InputType.WEIGHTS]))
    def test_threesigma(self, per_channel, input_type):
        collector = MedianMADStatisticCollector(per_channel, CHANNEL_AXIS, input_type)

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
            if input_type == InputType.INPUTS:
                samples = np.array([input_tensor for _ in range(step_num)])
            if input_type == InputType.WEIGHTS:
                samples = np.array([input_tensor])

            if per_channel:
                median = np.median(samples, axis=axis)
                mad = np.median(abs(samples - median))
                input_tensor_min_per_channel = (median - 3 * 1.4826 * mad).astype(np.float32)
                input_tensor_max_per_channel = (median + 3 * 1.4826 * mad).astype(np.float32)
            if not per_channel:
                inputs_tensor_flat = samples.flatten()
                median = np.median(inputs_tensor_flat)
                mad = np.median(abs(inputs_tensor_flat - median))
                input_tensor_min = (median - 3 * 1.4826 * mad).astype(np.float32)
                input_tensor_max = (median + 3 * 1.4826 * mad).astype(np.float32)

            if per_channel:
                self.run_all_checks(collector, step_num, per_channel, input_type,
                                    input_tensor_min_per_channel=input_tensor_min_per_channel,
                                    input_tensor_max_per_channel=input_tensor_max_per_channel)
            if not per_channel:
                self.run_all_checks(collector, step_num, per_channel, input_type,
                                    input_tensor_min=input_tensor_min,
                                    input_tensor_max=input_tensor_max)


@pytest.mark.parametrize("wrap_dataloader",
                         [True])
class TestPerLayerRangeInitTest:
    PerLayerRangeInitTestStruct = namedtuple('PerLayerRangeInitTestStruct',
                                             ('range_init_config',
                                              'layer_vs_expected_init_config'))

    qconfig = QuantizerConfig(num_bits=8,
                              mode=QuantizationMode.SYMMETRIC,
                              signedness_to_force=None,
                              per_channel=False)
    qspec = TFQuantizerSpec.from_config(qconfig, narrow_range=False, half_range=False)

    PER_LAYER_RANGE_INIT_TEST_CASES = [
        PerLayerRangeInitTestStruct(
            range_init_config=[{
                "type": "min_max",
                "num_init_samples": 1,
                "target_scopes": ["{re}.*"]
            }],
            layer_vs_expected_init_config=[
                (
                    (
                        NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv1")),
                        InputType.WEIGHTS
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1)
                ),
                (
                    (
                        FakeQuantize(qspec, name='fq1'),
                        InputType.INPUTS
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1)
                )]
        ),
        PerLayerRangeInitTestStruct(
            range_init_config=[{
                "type": "min_max",
                "num_init_samples": 1,
                "target_scopes": ["{re}conv.*"]
            }, {
                "type": "mean_min_max",
                "num_init_samples": 2,
                "ignored_scopes": ["{re}conv.*"]
            }],
            layer_vs_expected_init_config=[
                (
                    (
                        NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv1")),
                        InputType.WEIGHTS
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1)
                ),
                (
                    (
                        NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv2")),
                        InputType.WEIGHTS
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1)
                ),
                (
                    (
                        tf.keras.layers.Layer(name='conv2_0'),
                        InputType.INPUTS
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1)
                ),
                (
                    (
                        FakeQuantize(qspec, name='fq1'),
                        InputType.INPUTS
                    ),
                    RangeInitConfig(init_type="mean_min_max", num_init_samples=2)
                ),
            ]),
        PerLayerRangeInitTestStruct(
            range_init_config=[
                {
                    "type": "min_max",
                    "num_init_samples": 1,
                    "target_quantizer_group": "weights",
                    "target_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"]
                },
                {
                    "type": "mean_min_max",
                    "num_init_samples": 2,
                    "ignored_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*",
                                       "{re}/nncf_model_input_0"]
                },
                {
                    "type": "threesigma",
                    "num_init_samples": 1,
                    "target_quantizer_group": "activations",
                    "target_scopes": ["{re}/nncf_model_input_0"]
                },
                {
                    "type": "percentile",
                    "num_init_samples": 10,
                    "params": {
                        "min_percentile": "0.1",
                        "max_percentile": "99.9"
                    },
                    "target_quantizer_group": "activations",
                    "target_scopes": [
                        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0"]
                }
            ],
            layer_vs_expected_init_config=[
                (
                    (
                        tf.keras.layers.Layer(name='/nncf_model_input_0'),
                        InputType.INPUTS
                    ),
                    RangeInitConfig(init_type="threesigma", num_init_samples=1)
                ),
                (
                    (
                        tf.keras.layers.Layer(name="TwoConvTestModel/"
                                "Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0"),
                        InputType.WEIGHTS
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1)
                ),
                (
                    (
                        tf.keras.layers.Layer(name="TwoConvTestModel/"
                                "Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0"),
                        InputType.INPUTS
                    ),
                    RangeInitConfig(init_type="percentile", num_init_samples=10,
                                    init_type_specific_params={
                                        "min_percentile": "0.1",
                                        "max_percentile": "99.9"
                                    })
                ),
            ])
    ]

    @staticmethod
    @pytest.fixture(params=PER_LAYER_RANGE_INIT_TEST_CASES)
    def per_layer_range_init_test_struct(request):
        return request.param

    def test_get_init_config_for_quantization_point(self, wrap_dataloader, per_layer_range_init_test_struct):
        per_layer_configs = []
        for sub_init_range_config_dict in per_layer_range_init_test_struct.range_init_config:
            per_layer_configs.append(PerLayerRangeInitConfig.from_dict(sub_init_range_config_dict))

        params = TFRangeInitParams(wrap_dataloader,
                                   '',
                                   global_init_config=None,
                                   per_layer_range_init_configs=per_layer_configs)

        for ((layer, input_type), ref_range_init_config) in \
                per_layer_range_init_test_struct.layer_vs_expected_init_config:
            assert params.get_init_config_for_quantization_point(layer, input_type) == ref_range_init_config
