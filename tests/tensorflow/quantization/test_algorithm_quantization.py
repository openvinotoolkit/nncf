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
from tensorflow.python.keras import layers
import pytest

from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization import FakeQuantize
from nncf.tensorflow.quantization.algorithm import QuantizationController
from nncf.tensorflow.quantization.quantizers import Quantizer, TFQuantizerSpec
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.quantization.utils import get_basic_quantization_config
from nncf.common.quantization.structs import QuantizationMode


def compare_qspecs(qspec: TFQuantizerSpec, quantizer):
    assert qspec.num_bits == quantizer.num_bits
    assert qspec.per_channel == quantizer.per_channel
    assert qspec.narrow_range == quantizer.narrow_range
    assert qspec.half_range == quantizer.half_range
    assert isinstance(quantizer, NNCF_QUANTIZATION_OPERATONS.get(qspec.mode))
    if qspec.mode == QuantizationMode.SYMMETRIC:
        # pylint: disable=protected-access
        assert qspec.signedness_to_force == quantizer.signedness_to_force


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
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, should_init=False)
    assert isinstance(compression_ctrl, QuantizationController)

    activation_quantizers, weight_quantizers = get_quantizers(compression_model)
    for layer in compression_model.layers:
        if isinstance(layer, NNCFWrapper):
            for ops in layer.weights_attr_ops.values():
                for op in ops.values():
                    if isinstance(op, Quantizer):
                        weight_quantizers.append(op)

    ref_weight_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                       num_bits=8,
                                       signedness_to_force=None,
                                       per_channel=False,
                                       narrow_range=False,
                                       half_range=True)
    for wq in weight_quantizers:
        compare_qspecs(ref_weight_qspec, wq)

    ref_activation_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                           num_bits=8,
                                           signedness_to_force=None,
                                           per_channel=False,
                                           narrow_range=False,
                                           half_range=False)
    for wq in activation_quantizers:
        compare_qspecs(ref_activation_qspec, wq)


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
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, should_init=False)

    assert isinstance(compression_ctrl, QuantizationController)
    activation_quantizers, weight_quantizers = get_quantizers(compression_model)

    ref_weight_qspec = TFQuantizerSpec(mode=QuantizationMode.ASYMMETRIC,
                                       num_bits=4,
                                       signedness_to_force=None,
                                       per_channel=True,
                                       narrow_range=True,
                                       half_range=False)
    for wq in weight_quantizers:
        compare_qspecs(ref_weight_qspec, wq)

    ref_activation_qspec = TFQuantizerSpec(mode=QuantizationMode.ASYMMETRIC,
                                           num_bits=4,
                                           signedness_to_force=True,
                                           per_channel=False,
                                           narrow_range=False,
                                           half_range=False)
    for wq in activation_quantizers:
        compare_qspecs(ref_activation_qspec, wq)


def test_quantization_configs__disable_saturation_fix():
    model = get_basic_conv_test_model()

    config = get_basic_quantization_config()
    config['compression'].update({
        'disable_saturation_fix': True
    })
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, should_init=False)

    assert isinstance(compression_ctrl, QuantizationController)
    activation_quantizers, weight_quantizers = get_quantizers(compression_model)

    ref_weight_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                       num_bits=8,
                                       signedness_to_force=None,
                                       per_channel=False,
                                       narrow_range=True,
                                       half_range=False)
    for wq in weight_quantizers:
        compare_qspecs(ref_weight_qspec, wq)

    ref_activation_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                           num_bits=8,
                                           signedness_to_force=None,
                                           per_channel=False,
                                           narrow_range=False,
                                           half_range=False)
    for wq in activation_quantizers:
        compare_qspecs(ref_activation_qspec, wq)


@pytest.mark.parametrize('disabled', [False, True], ids=['enabled', 'disabled'])
def test_export_saturatuion_fix(disabled):
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    config['compression'].update({
        'disable_saturation_fix': disabled
    })
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, should_init=False)
    activation_quantizers_be, weight_quantizers_be = get_quantizers(compression_model)
    ref_weight_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                       num_bits=8,
                                       signedness_to_force=None,
                                       per_channel=False,
                                       narrow_range=disabled,
                                       half_range=not disabled)
    for wq in weight_quantizers_be:
        compare_qspecs(ref_weight_qspec, wq)

    ref_activation_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                           num_bits=8,
                                           signedness_to_force=None,
                                           per_channel=False,
                                           narrow_range=False,
                                           half_range=False)
    for wq in activation_quantizers_be:
        compare_qspecs(ref_activation_qspec, wq)

    compression_ctrl.export_model('/tmp/test.pb')
    activation_quantizers_ae, weight_quantizers_ae = get_quantizers(compression_model)

    ref_weight_qspec = TFQuantizerSpec(mode=QuantizationMode.SYMMETRIC,
                                       num_bits=8,
                                       signedness_to_force=None,
                                       per_channel=False,
                                       narrow_range=disabled,
                                       half_range=False)
    for wq in weight_quantizers_ae:
        compare_qspecs(ref_weight_qspec, wq)

    for wq in activation_quantizers_ae:
        compare_qspecs(ref_activation_qspec, wq)


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

    model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)
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

    model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)
    ref_fake_quantize_layers = ['input/fake_quantize']
    actual_fake_quantize_layers = [layer.name for layer in model.layers if isinstance(layer, FakeQuantize)]
    assert actual_fake_quantize_layers == ref_fake_quantize_layers
    assert len(actual_fake_quantize_layers) == len(ref_fake_quantize_layers)


class DataFormat:
    CF = 'channels_first'
    CL = 'channels_last'


LAYERS_PARAMS = {
    "Conv1D":
        {
            "param_name": ['filters', 'kernel_size', 'data_format'],
            "param_val": (8, 3,),
            "shape": (10, 32, 64),
        },
    "Conv2D":
        {
            "param_name": ['filters', 'kernel_size', 'data_format'],
            "param_val": (32, 3,),
            "shape": (10, 32, 64, 3),
        },
    "Conv3D":
        {
            "param_name": ['filters', 'kernel_size', 'data_format'],
            "param_val": (32, 3,),
            "shape": (10, 32, 64, 128, 3),
        },
    "DepthwiseConv2D":
        {
            "param_name": ['kernel_size', 'data_format'],
            "param_val": (32,),
            "shape": (10, 32, 64, 3),
        },
    "Conv1DTranspose":
        {
            "param_name": ['filters', 'kernel_size', 'data_format'],
            "param_val": (32, 32,),
            "shape": (10, 32, 64),
        },
    "Conv2DTranspose":
        {
            "param_name": ['filters', 'kernel_size', 'data_format'],
            "param_val": (32, 32,),
            "shape": (10, 32, 64, 3),
        },
    "Conv3DTranspose":
        {
            "param_name": ['filters', 'kernel_size', 'data_format'],
            "param_val": (32, 32,),
            "shape": (10, 8, 32, 64, 3),
        },
    "Dense":
        {
            "param_name": ['units'],
            "param_val": (32,),
            "shape": (10, 32),
        }
}


def get_layer_and_inputs(layer_name, params, shape, shape_build):
    layer = getattr(tf.keras.layers, layer_name)(**params)
    layer.build(shape_build)
    inputs = tf.reshape(tf.range(tf.reduce_prod(shape)), shape)
    return layer, inputs


class LayerDeck:
    def __init__(self, layer_name: str, input_type: InputType, data_format: DataFormat = None):
        self.layer_name = layer_name
        self.input_type = input_type
        self.inputs = None
        params = LAYERS_PARAMS[layer_name]
        self.layer_inputs_shape = params["shape"]
        self.params = dict()
        for k, v in zip(params["param_name"], params["param_val"]):
            self.params[k] = v
        if "data_format" in params["param_name"]:
            self.data_format = data_format
            self.params["data_format"] = data_format

    @property
    def shape(self):
        if self.inputs is not None:
            return tuple(self.inputs.shape)
        return None


class Conv1D(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Conv1D", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape[1:])

        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs_transformed = tf.transpose(self.inputs_transformed, [0, 2, 1])
            self.inputs_transformed = tf.reshape(self.inputs_transformed,
                                                 (-1, self.inputs_transformed.shape[-1]))

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.layer.weights[0]
            self.inputs_transformed = tf.reshape(self.inputs, (-1, self.params["filters"]))


class Conv2D(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Conv2D", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape[1:])
        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs_transformed = tf.transpose(self.inputs_transformed, [0, 3, 2, 1])

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.inputs_transformed = self.layer.weights[0]


class Conv3D(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Conv3D", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape[1:])
        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs_transformed = \
                                tf.transpose(self.inputs_transformed, [0, 4, 2, 3, 1])
            self.inputs_transformed = tf.reshape(self.inputs_transformed,
                                                 (-1, self.inputs_transformed.shape[-1]))

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.layer.weights[0]
            self.inputs_transformed = tf.reshape(self.inputs, (-1, self.params["filters"]))


class DepthwiseConv2D(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("DepthwiseConv2D", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape)
        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs_transformed = tf.transpose(self.inputs_transformed, [0, 3, 2, 1])

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.layer.weights[0]
            self.inputs_transformed = tf.reshape(self.inputs,
                                                 (-1, tf.math.reduce_prod(self.inputs.shape[2:])))


class Conv1DTranspose(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Conv1DTranspose", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape)

        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs_transformed = tf.transpose(self.inputs_transformed, [0, 2, 1])
            self.inputs_transformed = tf.reshape(self.inputs_transformed,
                                                 (-1, self.inputs_transformed.shape[-1]))

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.layer.weights[0]
            self.inputs_transformed = tf.transpose(self.inputs, [0, 2, 1])
            self.inputs_transformed = tf.reshape(self.inputs_transformed,
                                                 (-1, self.inputs_transformed.shape[-1]))


class Conv2DTranspose(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Conv2DTranspose", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape)
        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs = tf.transpose(self.inputs_transformed, [0, 3, 2, 1])

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.layer.weights[0]
            self.inputs_transformed = tf.transpose(self.inputs, [0, 1, 3, 2])


class Conv3DTranspose(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Conv3DTranspose", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape)
        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs
            if self.data_format == DataFormat.CF:
                self.inputs_transformed = tf.transpose(self.inputs_transformed, [0, 4, 2, 3, 1])
            self.inputs_transformed = tf.reshape(self.inputs_transformed,
                                                 (-1, self.inputs_transformed.shape[-1]))

        if self.input_type == InputType.WEIGHTS:
            self.inputs = self.layer.weights[0]
            self.inputs_transformed = tf.transpose(self.inputs, [0, 1, 2, 4, 3])
            self.inputs_transformed = tf.reshape(self.inputs_transformed,
                                                 (-1, self.inputs_transformed.shape[-1]))


class Dense(LayerDeck):
    def __init__(self, input_type: InputType, data_format: DataFormat):
        super().__init__("Dense", input_type, data_format)
        self.layer, self.inputs = get_layer_and_inputs(self.layer_name,
                                                       self.params,
                                                       self.layer_inputs_shape,
                                                       self.layer_inputs_shape[1:])
        if self.input_type == InputType.INPUTS:
            self.inputs_transformed = self.inputs

        if self.input_type == InputType.WEIGHTS:
            self.inputs_transformed = self.inputs = self.layer.weights[0]


LAYERS_MAP = {
    "Conv1D": Conv1D,
    "Conv2D": Conv2D,
    "Conv3D": Conv3D,
    "DepthwiseConv2D": DepthwiseConv2D,
    "Conv1DTranspose": Conv1DTranspose,
    "Conv2DTranspose": Conv2DTranspose,
    "Conv3DTranspose": Conv3DTranspose,
    "Dense": Dense,
}


def get_test_layers_desk():
    models = ["Conv1D", "Conv2D", "Conv3D", "DepthwiseConv2D",
              "Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose"]
    result = list()
    for model_name in models:
        for input_type in [InputType.INPUTS, InputType.WEIGHTS]:
            for data_format in [DataFormat.CF, DataFormat.CL]:
                result += [(model_name, input_type, data_format)]
    model_name = "Dense"
    for input_type in [InputType.INPUTS, InputType.WEIGHTS]:
        result += [(model_name, input_type, "")]
    return result


@pytest.mark.parametrize(
    'layer_name,input_type,data_type', get_test_layers_desk(), ids=[
        " ".join(l) for l in get_test_layers_desk()]
)
def test_quantize_pre_post_processing(layer_name, input_type, data_type):
    layer_desk = LAYERS_MAP[layer_name](input_type, data_type)
    layer_metatype = get_keras_layer_metatype(layer_desk.layer, determine_subtype=False)
    assert len(layer_metatype.weight_definitions) == 1
    layer_name = layer_metatype.weight_definitions[0].weight_attr_name
    q = Quantizer(name='quantizer')
    q.setup_input_transformation(layer_desk.shape, layer_desk.input_type,
                                 layer_name, layer_desk.layer)
    # pylint: disable=protected-access
    preprocess = q._pre_processing_fn(layer_desk.inputs)
    postprocess = q._post_processing_fn(preprocess)
    assert tf.math.reduce_all(preprocess == layer_desk.inputs_transformed)
    assert tf.math.reduce_all(postprocess == layer_desk.inputs)
