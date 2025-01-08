# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple

import pytest
import tensorflow as tf

from nncf.common.quantization.initialization.range import PerLayerRangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization import FakeQuantize
from nncf.tensorflow.quantization.init_range import TFRangeInitParams
from nncf.tensorflow.quantization.quantizers import AsymmetricQuantizer
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec


@pytest.mark.parametrize("wrap_dataloader", [True])
class TestPerLayerRangeInitTest:
    PerLayerRangeInitTestStruct = namedtuple(
        "PerLayerRangeInitTestStruct", ("range_init_config", "layer_vs_expected_init_config")
    )

    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False)
    qspec = TFQuantizerSpec.from_config(qconfig, narrow_range=False, half_range=False)

    PER_LAYER_RANGE_INIT_TEST_CASES = [
        PerLayerRangeInitTestStruct(
            range_init_config=[{"type": "min_max", "num_init_samples": 1, "target_scopes": ["{re}.*"]}],
            layer_vs_expected_init_config=[
                (
                    (NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv1")), InputType.WEIGHTS),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    (FakeQuantize(qspec, name="fq1"), InputType.INPUTS),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
            ],
        ),
        PerLayerRangeInitTestStruct(
            range_init_config=[
                {"type": "min_max", "num_init_samples": 1, "target_scopes": ["{re}conv.*"]},
                {"type": "mean_min_max", "num_init_samples": 2, "ignored_scopes": ["{re}conv.*"]},
            ],
            layer_vs_expected_init_config=[
                (
                    (NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv1")), InputType.WEIGHTS),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    (NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv2")), InputType.WEIGHTS),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    (tf.keras.layers.Layer(name="conv2_0"), InputType.INPUTS),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    (FakeQuantize(qspec, name="fq1"), InputType.INPUTS),
                    RangeInitConfig(init_type="mean_min_max", num_init_samples=2),
                ),
            ],
        ),
        PerLayerRangeInitTestStruct(
            range_init_config=[
                {
                    "type": "min_max",
                    "num_init_samples": 1,
                    "target_quantizer_group": "weights",
                    "target_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
                },
                {
                    "type": "mean_min_max",
                    "num_init_samples": 2,
                    "ignored_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*", "{re}/nncf_model_input_0"],
                },
                {
                    "type": "threesigma",
                    "num_init_samples": 1,
                    "target_quantizer_group": "activations",
                    "target_scopes": ["{re}/nncf_model_input_0"],
                },
                {
                    "type": "percentile",
                    "num_init_samples": 10,
                    "params": {"min_percentile": "0.1", "max_percentile": "99.9"},
                    "target_quantizer_group": "activations",
                    "target_scopes": ["TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0"],
                },
            ],
            layer_vs_expected_init_config=[
                (
                    (tf.keras.layers.Layer(name="/nncf_model_input_0"), InputType.INPUTS),
                    RangeInitConfig(init_type="threesigma", num_init_samples=1),
                ),
                (
                    (
                        tf.keras.layers.Layer(
                            name="TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0"
                        ),
                        InputType.WEIGHTS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    (
                        tf.keras.layers.Layer(
                            name="TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0"
                        ),
                        InputType.INPUTS,
                    ),
                    RangeInitConfig(
                        init_type="percentile",
                        num_init_samples=10,
                        init_type_specific_params={"min_percentile": "0.1", "max_percentile": "99.9"},
                    ),
                ),
            ],
        ),
    ]

    @staticmethod
    @pytest.fixture(params=PER_LAYER_RANGE_INIT_TEST_CASES)
    def per_layer_range_init_test_struct(request):
        return request.param

    def test_get_init_config_for_quantization_point(self, wrap_dataloader, per_layer_range_init_test_struct):
        per_layer_configs = []
        for sub_init_range_config_dict in per_layer_range_init_test_struct.range_init_config:
            per_layer_configs.append(PerLayerRangeInitConfig.from_dict(sub_init_range_config_dict))

        params = TFRangeInitParams(
            wrap_dataloader, "", global_init_config=None, per_layer_range_init_configs=per_layer_configs
        )

        for (
            layer,
            input_type,
        ), ref_range_init_config in per_layer_range_init_test_struct.layer_vs_expected_init_config:
            assert params.get_init_config_for_quantization_point(layer, input_type) == ref_range_init_config


@pytest.mark.parametrize(
    ("min_values", "max_values", "input_low", "input_range"),
    [
        ([0], [1], [0], [1]),
        ([1], [2], [0], [2]),
        ([-1], [0], [-1], [1]),
        ([-2], [-1], [-2], [2]),
        ([0], [0.1], [0], [0.1]),
        ([0], [0.05], [-0.025], [0.1]),
        ([0], [0.04], [-0.03], [0.1]),
        ([100], [101], [0], [101]),
        ([100], [200], [0], [200]),
        ([0, 1, -1, -2], [1, 2, 0, -1], [0, 0, -1, -2], [1, 2, 1, 2]),
        ([100, 100], [101, 200], [0, 0], [101, 200]),
        ([0, 0], [0.04, 100], [-0.48, 0], [1, 100]),
    ],
)
def test_asymmetric_apply_range_initialization(min_values, max_values, input_low, input_range):
    min_values_tf = tf.constant(min_values, dtype=tf.float32)
    max_values_tf = tf.constant(max_values, dtype=tf.float32)
    input_low_tf = tf.constant(input_low, dtype=tf.float32)
    input_range_tf = tf.constant(input_range, dtype=tf.float32)
    weights = dict(
        input_low_var=tf.Variable(tf.zeros_like(min_values_tf)),
        input_range_var=tf.Variable(tf.zeros_like(min_values_tf)),
    )

    asymmetric_quantizer = AsymmetricQuantizer("test_quantizer", TFQuantizerSpec(*((None,) * 6)))
    asymmetric_quantizer.apply_range_initialization(weights, min_values_tf, max_values_tf)

    assert tf.reduce_max(tf.abs(input_low_tf - weights["input_low_var"])).numpy() < 1e-5
    assert tf.reduce_max(tf.abs(input_range_tf - weights["input_range_var"])).numpy() < 1e-5
