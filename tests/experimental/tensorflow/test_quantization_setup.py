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

import os
import json

import tensorflow as tf

from tests.tensorflow import test_models
from tests.tensorflow.helpers import get_empty_config
from nncf.experimental.tensorflow.quantization.algorithm import QuantizationBuilder
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'target_points')


def test_quantization_setup():
    model = test_models.MobileNetV2(input_shape=(224, 224, 3))

    nncf_network = NNCFNetwork(
        model,
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='input_1')
    )

    config = get_empty_config(input_sample_sizes=(1, 224, 224, 3))
    config['compression'] = {
        'algorithm': 'quantization',
        'activations': {
            'per_channel': False
        },
        'weights': {
            'per_channel': False
        }
    }

    with open(f'{DATA_DIR}/MobileNetV2.json', encoding='utf8') as f:
        expected_qs = json.load(f)['target_points']

    builder = QuantizationBuilder(config, should_init=False)
    quantization_setup = builder.get_quantization_setup(nncf_network)
    actual_qs = [qp.op_name for qp in quantization_setup.get_quantization_points()]

    for x, y in zip(sorted(expected_qs), sorted(actual_qs)):
        assert x == y
