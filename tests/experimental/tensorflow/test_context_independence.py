"""
 Copyright (c) 2022 Intel Corporation
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

import os

import tensorflow as tf

from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.test_compressed_graph import create_test_name
from tests.tensorflow.test_compressed_graph import QuantizeTestCaseConfiguration
from tests.tensorflow.test_compressed_graph import get_basic_quantization_config
from tests.experimental.tensorflow.test_compressed_graph import check_model_graph_v2

from nncf.experimental.tensorflow.patch_tf import patch_tf_operations


patch_tf_operations()


class ModelWithSharedLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._conv = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')

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


def test_context_independence():
    params = {
        'activations': ('symmetric', 'per_tensor'),
        'weights': ('symmetric', 'per_tensor')
    }

    ref_graph_filename = 'simple.pb'
    graph_dir = os.path.join('quantized', create_test_name(params))
    case = QuantizeTestCaseConfiguration(params, graph_dir)
    input_sample_sizes = ([1, 28, 28, 1], [1, 28, 28, 1])
    config = get_basic_quantization_config(case.qconfig, input_sample_sizes)
    config['compression']['algorithm'] = 'experimental_quantization'
    models = []
    for _ in range(2):
        model = ModelWithSharedLayer()
        models.append(
            create_compressed_model_and_algo_for_test(model, config, force_no_init=True)[0]
        )

    for m in models:
        check_model_graph_v2(m, ref_graph_filename, graph_dir, False)
