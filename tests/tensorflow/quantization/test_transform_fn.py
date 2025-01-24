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
from tests.tensorflow.test_models.sequential_model import SequentialModel as ModelWithSingleInput


def ModelWithMultipleInputs():
    input_0 = tf.keras.Input(shape=(32, 32, 3))
    input_1 = tf.keras.Input(shape=(32, 32, 3))

    output_0 = tf.keras.layers.Conv2D(64, 3)(input_0)
    output_1 = tf.keras.layers.Conv2D(64, 3)(input_1)
    output = tf.keras.layers.Add()([output_0, output_1])
    return tf.keras.Model([input_0, input_1], output)


dataset = [
    {
        "input_0": tf.zeros((1, 32, 32, 3), dtype=tf.float32),
        "input_1": tf.zeros((1, 32, 32, 3), dtype=tf.float32),
    }
]


def single_input_transform_fn(data_item):
    return data_item["input_0"]


def multiple_inputs_transform_fn(data_item):
    return data_item["input_0"], data_item["input_1"]


@pytest.mark.parametrize(
    "model,transform_fn",
    [
        [ModelWithSingleInput(input_shape=(32, 32, 3)), single_input_transform_fn],
        [ModelWithMultipleInputs(), multiple_inputs_transform_fn],
    ],
    ids=[
        "single_input",
        "multiple_inputs",
    ],
)
def test_transform_fn(model, transform_fn):
    # Check the transformation function
    _ = model(transform_fn(next(iter(dataset))))

    # Start quantization
    calibration_dataset = nncf.Dataset(dataset, transform_fn)
    _ = nncf.quantize(model, calibration_dataset)
