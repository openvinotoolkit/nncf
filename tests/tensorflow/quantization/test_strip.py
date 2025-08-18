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

import pytest
import tensorflow as tf

import nncf
from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_two_conv_test_model
from tests.tensorflow.quantization.utils import get_basic_quantization_config
from tests.tensorflow.test_models.mobilenet_v2 import MobileNetV2


def test_strip():
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = tf.ones([1, 4, 4, 1])
    x_nncf = compressed_model(input_tensor)

    inference_model = compression_ctrl.strip()
    x_tf = inference_model(input_tensor)

    TFTensorListComparator.check_equal(x_nncf, x_tf)


@pytest.mark.parametrize("do_copy", (True, False))
def test_do_copy(do_copy):
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    inference_model = compression_ctrl.strip(do_copy=do_copy)

    if do_copy:
        assert id(inference_model) != id(compression_model)
    else:
        assert id(inference_model) == id(compression_model)


def test_strip_api():
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = tf.ones([1, 4, 4, 1])
    x_a = compressed_model(input_tensor)

    stripped_model = nncf.strip(compressed_model)
    x_b = stripped_model(input_tensor)

    TFTensorListComparator.check_equal(x_a, x_b)


@pytest.mark.parametrize("do_copy", (True, False))
def test_strip_api_do_copy(do_copy):
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    stripped_model = nncf.strip(compressed_model, do_copy=do_copy)

    if do_copy:
        assert id(stripped_model) != id(compressed_model)
    else:
        assert id(stripped_model) == id(compressed_model)


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._conv = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")

        self._bn_0 = tf.keras.layers.BatchNormalization()
        self._bn_1 = tf.keras.layers.BatchNormalization()
        self._add = tf.keras.layers.Add()
        self._flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        input_0, input_1 = inputs

        x_0 = self._conv(input_0)
        x_0 = self._bn_0(x_0, training=training)

        x_1 = self._conv(input_1)
        x_1 = self._bn_1(x_1, training=training)

        x_0 = self._flatten(x_0)
        x_1 = self._flatten(x_1)
        outputs = self._add([x_0, x_1])
        return outputs

    def get_config(self):
        raise NotImplementedError


def create_sequential_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, None, 3)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=2, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    return model


@pytest.mark.parametrize("model_fn", (MobileNetV2, SimpleModel, create_sequential_model))
def test_strip_api_no_compression(model_fn):
    model = model_fn()
    stripped_model = nncf.strip(model)
    assert id(stripped_model) == id(model)
