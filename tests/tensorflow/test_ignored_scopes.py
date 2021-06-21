"""
 Copyright (c) 2020 Intel Corporation
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

import tensorflow as tf
import tensorflow.keras.layers as layers

from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization import FakeQuantize
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.test_algorithm_quantization import get_basic_quantization_config


def test_ignored_scopes():
    config = get_basic_quantization_config(32)
    config['compression']['ignored_scopes'] = [
        'conv1',
        '{re}.*conv2.*'
    ]
    config['compression']['weights']['ignored_scopes'] = ['{re}.*conv3/c[23]']
    config['compression']['activations']['ignored_scopes'] = ['{re}.*c3$']

    model = tf.keras.Sequential([
        layers.Conv2D(3, 3, name='conv1', input_shape=config['input_info']['sample_size'][1:]),
        layers.Conv2D(3, 3, name='conv2'),
        layers.Conv2D(3, 3, name='conv2/c1'),
        layers.Conv2D(3, 3, name='some_scope/conv2/c1'),
        layers.Conv2D(3, 3, name='some_scope/conv3/c2'),
        layers.Conv2D(3, 3, name='some_scope/conv3/c3'),
        layers.Conv2D(3, 3, name='c3_1'),
        layers.Conv2D(3, 3, name='end')
    ])
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

    ref_fake_quantize_names = [
        'conv1_input/fake_quantize',
        'some_scope/conv3/c2/fake_quantize',
        'c3_1/fake_quantize'
    ]
    ref_nncf_wrapper_names = [
        'c3_1',
        'end'
    ]

    fake_quantize_names = [layer.name for layer in compressed_model.layers if isinstance(layer, FakeQuantize)]
    nncf_wrapper_names = [layer.name for layer in compressed_model.layers if isinstance(layer, NNCFWrapper)]
    assert fake_quantize_names == ref_fake_quantize_names
    assert nncf_wrapper_names == ref_nncf_wrapper_names
