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

from addict import Dict

from tensorflow.python.keras import layers
import tensorflow as tf
import numpy as np
import pytest

from beta.nncf import NNCFConfig
from beta.tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers


def get_basic_pruning_config(model_size=8):
    config = NNCFConfig()
    config.update(Dict({
        "model": "basic",
        "input_info":
            {
                "sample_size": [1, model_size, model_size, 1],
            },
        "compression":
            {
                "algorithm": "filter_pruning",
                "pruning_init": 0.5,
                "params": {
                    "prune_first_conv": True,
                    "prune_last_conv": True
                }
            }
    }))
    return config


def check_pruning_mask(mask, pruning_rate, layer_name):
    assert np.sum(mask) == mask.size * pruning_rate, f"Incorrect masks for {layer_name}"


def get_concat_test_model(input_shape):
    #            (input)
    #               |
    #            (conv1)
    #        /      |      \
    #    (conv2) (conv3) (conv4 + bn)
    #       |       |       |
    #        \      |      /
    #          (tf_concat)
    #               |
    #          (BatchNorm)
    #               |
    #            (conv5)

    inputs = tf.keras.Input(shape=input_shape[1:], name='input')
    conv1 = layers.Conv2D(16, 1, name='conv1')
    conv2 = layers.Conv2D(16, 1, name='conv2')
    conv3 = layers.Conv2D(16, 1, name='conv3')
    conv4 = layers.Conv2D(32, 1, name='conv4')
    bn_conv4 = layers.BatchNormalization(name="bn_conv4")
    bn_concat = layers.BatchNormalization(name="bn_concat")
    conv5 = layers.Conv2D(48, 1, name='conv5')

    x = conv1(inputs)
    x1 = tf.concat([conv2(x), conv3(x)], -1, name='tf_concat_1')
    x1 = conv4(x1)
    x1 = bn_conv4(x1)
    x = tf.concat([x, x1], -1, name='tf_concat_2')
    x = bn_concat(x)
    outputs = conv5(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


@pytest.mark.parametrize(('all_weights', 'prune_batch_norms', 'ref_num_wrapped_layer'),
                         [
                             [True, True, 7],
                             [False, True, 7],
                             [True, False, 5],
                             [False, False, 5],
                         ])
def test_masks_in_concat_model(all_weights, prune_batch_norms, ref_num_wrapped_layer):
    config = get_basic_pruning_config(8)
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['params']['prune_batch_norms'] = prune_batch_norms
    sample_size = [1, 8, 8, 3]
    model = get_concat_test_model(sample_size)

    model, _ = create_compressed_model_and_algo_for_test(model, config)
    wrapped_layers = collect_wrapped_layers(model)

    # Check number of wrapped layers
    assert len(wrapped_layers) == ref_num_wrapped_layer

    for layer in wrapped_layers:
        # Check existed weights of masks
        assert layer.ops_weights

        # Check masks correctness
        if not all_weights:
            for op in layer.ops_weights.values():
                check_pruning_mask(op['mask'].numpy(), 0.5, layer.name)
