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

import tensorflow as tf

from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.helpers import get_basic_conv_test_model


def test_basic_model_has_expected_params():
    default_weight = tf.constant([[[[0.0, -1.0], [-1.0, 0.0]]], [[[0.0, -1.0], [-1.0, 0.0]]]])
    default_weight = tf.transpose(default_weight, (2, 3, 1, 0))
    default_bias = tf.constant([-2.0, -2.0])
    model = get_basic_conv_test_model()
    act_weights = model.layers[1].weights[0]
    ref_weights = default_weight
    act_bias = model.layers[1].weights[1]
    ref_bias = default_bias

    TFTensorListComparator.check_equal(act_bias, ref_bias)
    TFTensorListComparator.check_equal(act_weights, ref_weights)


def test_basic_model_is_valid():
    model = get_basic_conv_test_model()
    input_ = tf.ones([1, 4, 4, 1])
    ref_output = tf.ones((1, 3, 3, 2)) * (-4)
    act_output = model(input_)
    TFTensorListComparator.check_equal(ref_output, act_output)
