# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tensorflow as tf

from tests.tensorflow import test_models
from tests.tensorflow.test_compressed_graph import keras_model_to_tf_graph
from tests.tensorflow.test_compressed_graph import rename_graph_def_nodes


def get_graph(model):
    model_from_graph, graph_to_layer_var_names_map = keras_model_to_tf_graph(model)
    graph_def = model_from_graph.as_graph_def(add_shapes=True)
    rename_graph_def_nodes(graph_def, graph_to_layer_var_names_map)
    return graph_def


def get_test_model_pairs():
    return [
        (test_models.DenseNet121, tf.keras.applications.densenet.DenseNet121),
        (test_models.InceptionResNetV2, tf.keras.applications.InceptionResNetV2),
        (test_models.InceptionV3, tf.keras.applications.inception_v3.InceptionV3),
        (test_models.MobileNet, tf.keras.applications.mobilenet.MobileNet),
        (test_models.MobileNetV2, tf.keras.applications.mobilenet_v2.MobileNetV2),
        (test_models.NASNetMobile, tf.keras.applications.nasnet.NASNetMobile),
        (test_models.ResNet50, tf.keras.applications.ResNet50),
        (test_models.ResNet50V2, tf.keras.applications.resnet_v2.ResNet50V2),
        (test_models.VGG16, tf.keras.applications.vgg16.VGG16),
        (test_models.Xception, tf.keras.applications.xception.Xception),
        pytest.param(
            test_models.MobileNetV3Small,
            tf.keras.applications.MobileNetV3Small,
            marks=pytest.mark.skip(reason="model definition differ"),
        ),
        pytest.param(
            test_models.MobileNetV3Large,
            tf.keras.applications.MobileNetV3Large,
            marks=pytest.mark.skip(reason="model definition differ"),
        ),
    ]


@pytest.mark.parametrize("nncf_model_builder,keras_application_model_builder", get_test_model_pairs())
def test_nncf_keras_model_graph_equality(nncf_model_builder, keras_application_model_builder):
    nncf_model = nncf_model_builder()
    graph_def_nncf = get_graph(nncf_model)

    tf.keras.backend.clear_session()

    keras_application_model = keras_application_model_builder(weights=None)
    graph_def_keras = get_graph(keras_application_model)

    tf.test.assert_equal_graph_def(graph_def_nncf, graph_def_keras)
