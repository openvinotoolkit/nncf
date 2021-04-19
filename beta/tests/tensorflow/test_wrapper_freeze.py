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

from beta.tests.tensorflow.helpers import get_basic_conv_test_model
from beta.nncf import NNCFConfig
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.tests.tensorflow.helpers import create_compressed_model_and_algo_for_test


def get_basic_quantization_config(model_size=4):
    config = NNCFConfig()
    config.update(Dict({
        "model": "basic_quant_conv",
        "input_info":
            {
                "sample_size": [1, model_size, model_size, 1],
            },
        "compression":
            {
                "algorithm": "quantization",
            }
    }))
    return config


def check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights):
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            actual_trainable_weights = [weight.name for weight in layer.trainable_weights]
            actual_non_trainable_weights = [weight.name for weight in layer.non_trainable_weights]
            assert actual_trainable_weights == ref_trainable_weights
            assert actual_non_trainable_weights == ref_non_trainable_weights


def test_wrapper_weights_freeze():
    config = get_basic_quantization_config()
    model = get_basic_conv_test_model()
    model, _ = create_compressed_model_and_algo_for_test(model, config)

    ref_trainable_weights = ['nncf_wrapper_conv2d/kernel_scale:0', 'nncf_wrapper_conv2d/kernel:0',
                             'nncf_wrapper_conv2d/bias:0']
    ref_non_trainable_weights = ['nncf_wrapper_conv2d/kernel_signed:0']
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.trainable = False
    ref_trainable_weights = []
    ref_non_trainable_weights = ['nncf_wrapper_conv2d/kernel_signed:0', 'nncf_wrapper_conv2d/bias:0',
                                 'nncf_wrapper_conv2d/kernel_scale:0', 'nncf_wrapper_conv2d/kernel:0']
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.set_ops_trainable(True)
    ref_trainable_weights = ['nncf_wrapper_conv2d/kernel_scale:0']
    ref_non_trainable_weights = ['nncf_wrapper_conv2d/kernel_signed:0', 'nncf_wrapper_conv2d/bias:0',
                                 'nncf_wrapper_conv2d/kernel:0']
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.trainable = True
    ref_trainable_weights = ['nncf_wrapper_conv2d/kernel_scale:0', 'nncf_wrapper_conv2d/kernel:0',
                             'nncf_wrapper_conv2d/bias:0']
    ref_non_trainable_weights = ['nncf_wrapper_conv2d/kernel_signed:0']
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.set_ops_trainable(False)
    ref_trainable_weights = ['nncf_wrapper_conv2d/kernel:0', 'nncf_wrapper_conv2d/bias:0']
    ref_non_trainable_weights = ['nncf_wrapper_conv2d/kernel_signed:0', 'nncf_wrapper_conv2d/kernel_scale:0']
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)
