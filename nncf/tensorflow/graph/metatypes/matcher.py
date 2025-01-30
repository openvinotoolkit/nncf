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

from typing import Type

import tensorflow as tf

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.tensorflow.graph.metatypes.keras_layers import KERAS_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.tf_ops import TF_OPERATION_METATYPES


def get_keras_layer_metatype(layer: tf.keras.layers.Layer, determine_subtype: bool = True) -> Type[OperatorMetatype]:
    """
    Returns a metatype of the Keras layer.

    The flag 'determine_subtype' specifies which metatype for the layer the subtype
    or main type will be returned.

    For example, you created instance of the depthwise convolution using
    `tf.keras.layers.Conv2D`, then `get_keras_layer_metatype` will return different
    metatypes depending on the determine_subtype flag.

    ```
    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)
    layer = tf.keras.layers.Conv2D(3, 3, groups = 3, input_shape=input_shape[1:])
    y = layer(x)

    metatype = get_keras_layer_metatype(layer, determine_subtype = False)
    assert metatype == TFConv2DLayerMetatype
    metatype = get_keras_layer_metatype(layer, determine_subtype = True)
    assert metatype == TFDepthwiseConv2DSubLayerMetatype
    ```

    :param layer: The Keras layer.
    :param determine_subtype: Determines the subtype of the metatype if True and
        returns the primary metatype otherwise.
    :return: A metatype.
    """
    layer_metatype = KERAS_LAYER_METATYPES.get_operator_metatype_by_op_name(layer.__class__.__name__)

    if not determine_subtype:
        return layer_metatype
    subtype = None
    if layer_metatype is not UnknownMetatype:
        subtype = layer_metatype.determine_subtype(layer)
    if subtype is not None:
        return subtype
    return layer_metatype


def get_op_metatype(op_name: str) -> Type[OperatorMetatype]:
    """
    Returns a metatype of the TF operation by operation name.

    :param op_name: TF operation name.
    :return: A metatype.
    """
    return TF_OPERATION_METATYPES.get_operator_metatype_by_op_name(op_name)
