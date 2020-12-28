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

from addict import Dict

import tensorflow as tf
from tensorflow.python.keras import layers

from tests.tensorflow.helpers import get_basic_conv_test_model, create_compressed_model_and_algo_for_test
from nncf.tensorflow.quantization import FakeQuantize
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.algorithm import QuantizationController
from nncf.tensorflow.quantization.config import QuantizerConfig, QuantizationMode
from nncf import Config


def get_basic_quantization_config(model_size=4):
    config = Config()
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


def get_basic_asym_quantization_config(model_size=4):
    config = get_basic_quantization_config(model_size)
    config['compression']['activations'] = {'mode': 'asymmetric'}
    return config


def compare_qconfigs(config: QuantizerConfig, quantizer):
    assert config.num_bits == quantizer.num_bits
    assert config.per_channel == quantizer.per_channel
    assert config.narrow_range == quantizer.narrow_range
    assert isinstance(quantizer, NNCF_QUANTIZATION_OPERATONS.get(config.mode))
    if config.mode == QuantizationMode.SYMMETRIC:
        # pylint: disable=protected-access
        assert config.signed == quantizer._initial_signedness


def get_quantizers(model):
    # pylint: disable=protected-access
    activation_quantizers = [layer._quantizer for layer in model.layers
                             if isinstance(layer, FakeQuantize)]
    weight_quantizers = []
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            for ops in layer.weights_attr_ops.values():
                for op in ops.values():
                    if isinstance(op, Quantizer):
                        weight_quantizers.append(op)
    return activation_quantizers, weight_quantizers


def test_quantization_configs__with_defaults():
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, QuantizationController)

    activation_quantizers, weight_quantizers = get_quantizers(compression_model)
    for layer in compression_model.layers:
        if isinstance(layer, NNCFWrapper):
            for ops in layer.weights_attr_ops.values():
                for op in ops.values():
                    if isinstance(op, Quantizer):
                        weight_quantizers.append(op)

    ref_weight_qconfig = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                         num_bits=8,
                                         signed=None,
                                         per_channel=False,
                                         narrow_range=True)
    for wq in weight_quantizers:
        compare_qconfigs(ref_weight_qconfig, wq)

    ref_activation_qconfig = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                             num_bits=8,
                                             signed=None,
                                             per_channel=False,
                                             narrow_range=False)
    for wq in activation_quantizers:
        compare_qconfigs(ref_activation_qconfig, wq)


def test_quantization_configs__custom():
    model = get_basic_conv_test_model()

    config = get_basic_quantization_config()
    config['compression'].update({
        "weights": {
            "mode": "asymmetric",
            "per_channel": True,
            "bits": 4
        },
        "activations": {
            "mode": "asymmetric",
            "bits": 4,
            "signed": True,
        },
    })
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    activation_quantizers, weight_quantizers = get_quantizers(compression_model)

    ref_weight_qconfig = QuantizerConfig(mode=QuantizationMode.ASYMMETRIC,
                                         num_bits=4,
                                         signed=None,
                                         per_channel=True,
                                         narrow_range=True)
    for wq in weight_quantizers:
        compare_qconfigs(ref_weight_qconfig, wq)

    ref_activation_qconfig = QuantizerConfig(mode=QuantizationMode.ASYMMETRIC,
                                             num_bits=4,
                                             signed=True,
                                             per_channel=False,
                                             narrow_range=False)
    for wq in activation_quantizers:
        compare_qconfigs(ref_activation_qconfig, wq)


def get_quantize_inputs_test_model(input_shapes):
    #    (1)     (2)      (3)    (4)   (5)
    #     |       |        |      |     |-----\
    #  (conv1)   (MP)     (MP)    (MP)  (MP)  |
    #     |       |       | |     |     |     |
    #     |       |       (+)     |     |     |
    #     |       |--\     |      |     |     |
    #     |       |   \    |      |     |     |
    #     |    (conv2) | (conv3)  |     |     |
    #     |       |    |   |       \   /      |
    #     |     (AvP)  \   |       (cat)      |
    #     |       |     \  |         |        |
    #  (conv4) (linear)  \ |      (conv6)     |
    #     |       |      (cat)       |        |
    #     |       |        |        (+)------/
    #     |       |      (conv5)     |
    #   (AvP)     |        |         |
    #     |       |      (AvP)       |
    #      \      |        /         |
    #       \---(cat)---------------/
    #             |
    #           (dense)

    inputs = []
    for i, input_shape in enumerate(input_shapes):
        inputs.append(tf.keras.Input(shape=input_shape[1:], name='input_{}'.format(i + 1)))
    # pylint: disable=unbalanced-tuple-unpacking
    input_1, input_2, input_3, input_4, input_5 = inputs

    conv1 = layers.Conv2D(filters=8, kernel_size=3)
    conv2 = layers.Conv2D(filters=8, kernel_size=3)
    conv3 = layers.Conv2D(filters=8, kernel_size=3)
    conv4 = layers.Conv2D(filters=16, kernel_size=3)
    conv5 = layers.Conv2D(filters=3, kernel_size=1)
    conv6 = layers.Conv2D(filters=3, kernel_size=2)
    dense = layers.Dense(8)

    x_1 = layers.Rescaling(1. / 255.)(input_1)
    x_1 = conv1(x_1)
    x_1 = conv4(x_1)
    x_1 = layers.GlobalAveragePooling2D()(x_1)
    x_1 = layers.Flatten()(x_1)

    x_2_br = layers.MaxPool2D(pool_size=2)(input_2)
    x_2 = conv2(x_2_br)
    x_2 = layers.GlobalAveragePooling2D()(x_2)
    x_2 = layers.Flatten()(x_2)
    x_2 = dense(x_2)

    x_3 = layers.MaxPool2D(pool_size=2)(input_3)
    x_3 = x_3 + x_3
    x_3 = conv3(x_3)
    x_3 = layers.Flatten()(x_3)
    x_2_br = layers.Flatten()(x_2_br)
    x_3 = tf.concat([x_2_br, x_3], -1)
    x_3 = conv5(tf.expand_dims(tf.expand_dims(x_3, -1), -1))
    x_3 = layers.GlobalAveragePooling2D()(x_3)
    x_3 = layers.Flatten()(x_3)

    x_4 = layers.MaxPool2D(pool_size=2)(input_4)
    x_5 = layers.MaxPool2D(pool_size=2)(input_5)
    x_45 = tf.concat([x_4, x_5], -1)
    x_45 = conv6(x_45)
    x_45 = layers.Flatten()(x_45)
    in_5_flat = layers.Flatten()(input_5)
    # pylint: disable=E1120
    x_45 = tf.pad(x_45, [[0, 0], [0, in_5_flat.shape[1] - x_45.shape[1]]])
    x_45 += in_5_flat
    x = tf.concat([x_1, x_2, x_3, x_45], -1)
    outputs = layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_quantize_inputs():
    config = get_basic_quantization_config()
    input_shapes = [[2, 32, 32, 3] for i in range(5)]
    model = get_quantize_inputs_test_model(input_shapes)

    model, _ = create_compressed_model_and_algo_for_test(model, config)
    ref_fake_quantize_layers_for_inputs ={
        'rescaling/fake_quantize',
        'input_2/fake_quantize',
        'input_3/fake_quantize',
        'input_4/fake_quantize',
        'input_5/fake_quantize'
    }
    ref_fake_quantize_layers = 17

    actual_fake_quantize_layers = {layer.name for layer in model.layers if isinstance(layer, FakeQuantize)}
    assert ref_fake_quantize_layers_for_inputs.issubset(actual_fake_quantize_layers)
    assert len(actual_fake_quantize_layers) == ref_fake_quantize_layers


def get_quantize_outputs_removal_test_model(input_shape):
    #               (input)
    #       /      /      \     \
    #   (conv1) (conv2) (conv3) (conv4)
    #     |        |         \        \
    #     |        |          \        \
    # (flatten1) (flatten1) (flatten2) (GlobalMaxPool2D)
    #        \      |       /           /
    #          (keras_concat)         /
    #                      \        /
    #                     (tf_concat)
    #                         /
    #                 (keras_reshape)
    #                    /
    #             (tf_reshape)

    inputs = tf.keras.Input(shape=input_shape[1:], name='input')
    conv1 = layers.Conv2D(3, 8, name='conv1')
    conv2 = layers.Conv2D(3, 8, name='conv2')
    conv3 = layers.Conv2D(3, 8, name='conv3')
    conv4 = layers.Conv2D(3, 8, name='conv4')
    flatten1 = layers.Flatten()
    flatten2 = layers.Flatten()
    keras_concat = layers.Concatenate(name='keras_concat')
    x1 = conv1(inputs)
    x1 = flatten1(x1)
    x2 = conv2(inputs)
    x2 = flatten1(x2)
    x3 = conv3(inputs)
    x3 = flatten2(x3)
    x4 = conv4(inputs)
    x4 = layers.GlobalMaxPool2D()(x4)
    x123 = keras_concat([x1, x2, x3])
    x = tf.concat([x123, x4], -1, name='tf_concat')
    x = layers.Reshape((-1, 4), name='keras_reshape')(x)
    outputs = tf.reshape(x, (-1, 2, 2), name='tf_reshape')
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_quantize_outputs_removal():
    config = get_basic_quantization_config()
    sample_size = [2, 32, 32, 3]
    model = get_quantize_outputs_removal_test_model(sample_size)

    model, _ = create_compressed_model_and_algo_for_test(model, config)
    ref_fake_quantize_layers = ['input/fake_quantize']
    actual_fake_quantize_layers = [layer.name for layer in model.layers if isinstance(layer, FakeQuantize)]
    assert actual_fake_quantize_layers == ref_fake_quantize_layers
    assert len(actual_fake_quantize_layers) == len(ref_fake_quantize_layers)
