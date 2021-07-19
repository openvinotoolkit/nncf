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

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.graph import OUTPUT_NOOP_METATYPES
from nncf.tensorflow.graph.converter import TFModelConverter
from nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.test_algorithm_quantization import get_basic_quantization_config


def test_struct_auxiliary_nodes_nncf_graph():
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)

    nncf_graph = convert_keras_model_to_nncf_graph(compressed_model)

    input_nodes = nncf_graph.get_input_nodes()
    output_nodes = nncf_graph.get_output_nodes()

    assert len(input_nodes) == 1
    assert len(output_nodes) == 1

    assert input_nodes[0].metatype in INPUT_NOOP_METATYPES
    assert output_nodes[0].metatype in OUTPUT_NOOP_METATYPES


class CustomLayerForTest(tf.keras.layers.Layer):
    CUSTOM_LAYER_NAME = "custom_layer_for_test"

    def __init__(self):
        super().__init__(name=self.CUSTOM_LAYER_NAME)
        self.w = self.add_weight(shape=(1, ))

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, self.w)


def ModelForCustomLayerTest():
    input_shape = (None, None, 3)
    img_input = layers.Input(shape=input_shape) # non-custom
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = img_input
    x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(x)  # non-custom, but experimental
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)  # non-custom
    x = CustomLayerForTest()(x)  # custom!
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)  # non-custom
    x = tf.multiply(x, x)  # TensorFlowOpLayer, should be treated as non-custom

    model = models.Model(img_input, x, name='ModelForCustomLayerTest')
    return model


def test_get_custom_layers():
    model = ModelForCustomLayerTest()
    model.build([16, 16, 3])
    custom_layers = TFModelConverter.get_custom_layers(model)
    assert len(custom_layers) == 1
    assert CustomLayerForTest.CUSTOM_LAYER_NAME in custom_layers
    assert isinstance(custom_layers[CustomLayerForTest.CUSTOM_LAYER_NAME], CustomLayerForTest)
