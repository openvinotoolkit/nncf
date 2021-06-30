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
from copy import deepcopy

import tensorflow as tf

from nncf import NNCFConfig
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.initialization import register_default_init_args


def get_dataset_for_test(batch_size=10):
    rand_image = tf.random.uniform(shape=[5, 5, 1], dtype=tf.float32)
    dataset1 = tf.data.Dataset.from_tensors(rand_image)
    rand_label = tf.random.uniform(shape=[], dtype=tf.float32)
    dataset2 = tf.data.Dataset.from_tensors(rand_label)
    dataset = tf.data.Dataset.zip((dataset1, dataset2)).repeat(100).batch(batch_size)
    return dataset


def get_config_for_test(batch_size=10, num_bn_adaptation_samples=100, num_bn_forget_samples=50):
    config = NNCFConfig()
    config.update(Dict({
        "compression":
            {
                "initializer": {
                    "batchnorm_adaptation": {
                        "num_bn_adaptation_samples": num_bn_adaptation_samples,
                        "num_bn_forget_samples": num_bn_forget_samples
                    }
                }
            }
    }))

    dataset = get_dataset_for_test()
    config = register_default_init_args(config,
                                        dataset,
                                        batch_size)

    compression_config = config.get('compression', {})
    compression_config = NNCFConfig(compression_config)
    compression_config.register_extra_structs(config.get_all_extra_structs_for_copy())

    return compression_config


def get_model_for_test():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(5, 5, 1)),
            tf.keras.layers.Conv2D(2, 3, name="layer1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(4, name="layer2")
        ]
    )
    return model


def compare_params(weights_a, weights_b, equal=True):
    for i, _ in enumerate(weights_a):
        if equal:
            assert tf.reduce_all(tf.equal(weights_a[i], weights_b[i]))
        else:
            assert tf.reduce_all(tf.not_equal(weights_a[i], weights_b[i]))


def test_parameter_update():
    original_param_values = {}
    original_bn_stat_values = {}

    model = get_model_for_test()

    for layer in model.layers:
        if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
            original_bn_stat_values[layer] = deepcopy(layer.non_trainable_weights)
            original_param_values[layer] = deepcopy(layer.trainable_weights)
        else:
            original_param_values[layer] = deepcopy(layer.weights)

    config = get_config_for_test()

    bn_adaptation = BatchnormAdaptationAlgorithm(**extract_bn_adaptation_init_params(config))
    bn_adaptation.run(model)

    for layer in model.layers:
        if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
            compare_params(original_bn_stat_values[layer], layer.non_trainable_weights, equal=False)
            compare_params(original_param_values[layer], layer.trainable_weights)
        else:
            compare_params(original_param_values[layer], layer.weights)


def test_all_parameter_keep():
    original_all_param_values = {}

    model = get_model_for_test()

    for layer in model.layers:
        original_all_param_values[layer] = deepcopy(layer.weights)

    config = get_config_for_test(num_bn_adaptation_samples=0, num_bn_forget_samples=0)

    bn_adaptation = BatchnormAdaptationAlgorithm(**extract_bn_adaptation_init_params(config))
    bn_adaptation.run(model)

    for layer in model.layers:
        compare_params(original_all_param_values[layer], layer.weights)
