"""
 Copyright (c) 2023 Intel Corporation
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
from collections import Counter

import pytest
import tensorflow as tf
from tensorflow.keras import layers

from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization import FakeQuantize
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.test_algorithm_quantization import get_basic_quantization_config
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.helpers import get_mock_model


def test_ignored_scopes():
    config = get_basic_quantization_config(32)
    config['compression']['ignored_scopes'] = [
        'conv1',
        '{re}.*conv2.*'
    ]
    config['compression']['weights'] = {'ignored_scopes': ['{re}.*conv3/c[23]']}
    config['compression']['activations'] = {'ignored_scopes': ['{re}.*c3$']}

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

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)

    ref_fake_quantize_names = [
        'some_scope/conv2/c1/fake_quantize',
        'some_scope/conv3/c3/fake_quantize',
        'c3_1/fake_quantize'
    ]
    ref_nncf_wrapper_names = [
        'c3_1',
        'end'
    ]

    fake_quantize_names = [layer.name for layer in compressed_model.layers if isinstance(layer, FakeQuantize)]
    nncf_wrapper_names = [layer.name for layer in compressed_model.layers if isinstance(layer, NNCFWrapper)]
    assert Counter(fake_quantize_names) == Counter(ref_fake_quantize_names)
    assert Counter(nncf_wrapper_names) == Counter(ref_nncf_wrapper_names)


NOT_SUPPORT_SCOPES_ALGO = ["NoCompressionAlgorithm"]
@pytest.mark.parametrize("algo_name", TF_COMPRESSION_ALGORITHMS.registry_dict.keys() - NOT_SUPPORT_SCOPES_ALGO)
def test_raise_runtimeerror_for_not_matched_scope_names(algo_name):
    model = get_mock_model()
    config = get_empty_config()
    config['compression'] = {'algorithm': algo_name, 'ignored_scopes': ['unknown']}
    if algo_name == "quantization":
        config['compression']["initializer"] = {"batchnorm_adaptation": {"num_bn_adaptation_samples": 0}}

    with pytest.raises(RuntimeError) as exc_info:
        create_compressed_model_and_algo_for_test(model, config)
    assert "No match has been found among the model" in str(exc_info.value)
