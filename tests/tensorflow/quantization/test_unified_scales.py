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

import itertools

import pytest
import tensorflow as tf
from tensorflow.keras import layers

from nncf.common.hardware.config import HWConfigType
from nncf.tensorflow.quantization.utils import collect_fake_quantize_layers
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.utils import get_basic_quantization_config


def get_single_concat_test_model(input_shapes):
    inputs = []
    for i, input_shape in enumerate(input_shapes):
        inputs.append(tf.keras.Input(shape=input_shape[1:], name="input_{}".format(i + 1)))

    input_1, input_2 = inputs

    x_1 = layers.Multiply()([input_1, input_1])
    x_2 = layers.Multiply()([input_2, input_2])

    outputs = layers.Concatenate(1)([x_1, x_2])
    outputs = layers.Conv2D(filters=1, kernel_size=1)(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_double_concat_test_model(input_shapes):
    inputs = []
    for i, input_shape in enumerate(input_shapes):
        inputs.append(tf.keras.Input(shape=input_shape[1:], name="input_{}".format(i + 1)))

    input_1, input_2 = inputs

    x_1 = input_1 * input_1
    x_2 = input_2 * input_2

    cat_1 = layers.Concatenate(1)([x_1, x_2])
    cat_2 = layers.Concatenate(1)([x_1, cat_1])
    outputs = layers.Conv2D(filters=3, kernel_size=3, strides=2, padding="same")(cat_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_unet_like_test_model(input_shapes):
    inputs = []
    for i, input_shape in enumerate(input_shapes):
        inputs.append(tf.keras.Input(shape=input_shape[1:], name="input_{}".format(i + 1)))

    input_1, _ = inputs

    conv_1 = layers.Conv2D(filters=8, kernel_size=1)(input_1)
    conv_2 = layers.Conv2D(filters=16, kernel_size=1)(conv_1)
    conv_3 = layers.Conv2D(filters=32, kernel_size=1)(conv_2)
    conv_t_3 = layers.Conv2DTranspose(filters=16, kernel_size=1)(conv_3)

    cat_1 = layers.Concatenate(0)([conv_t_3, conv_2])
    conv_t_2 = layers.Conv2DTranspose(filters=8, kernel_size=1)(cat_1)

    cat_2 = layers.Concatenate(0)([conv_t_2, conv_1])
    outputs = layers.Conv2DTranspose(filters=4, kernel_size=1)(cat_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


CAT_UNIFIED_SCALE_TEST_STRUCTS = [
    (get_single_concat_test_model, 1, 2),
    (get_double_concat_test_model, 1, 2),
    (get_unet_like_test_model, 4, 6),
]


def get_total_quantizations(model: tf.keras.Model) -> int:
    fq_layers = [layer for layer in model.get_config()["layers"] if layer["class_name"] == "FakeQuantize"]
    total_quantizations = sum(len(layer["inbound_nodes"]) for layer in fq_layers)
    return total_quantizations


@pytest.mark.parametrize(
    "target_device, model_creator, ref_aq_module_count, ref_quantizations",
    [
        (t_dev,) + rest
        for t_dev, rest in itertools.product([x.value for x in HWConfigType], CAT_UNIFIED_SCALE_TEST_STRUCTS)
    ],
)
def test_unified_scales_with_concat(target_device, model_creator, ref_aq_module_count, ref_quantizations):
    nncf_config = get_basic_quantization_config()
    x_shape = [1, 4, 1, 1]
    y_shape = [1, 4, 1, 1]
    nncf_config["target_device"] = target_device

    model = model_creator([x_shape, y_shape])
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config, force_no_init=True)
    non_weight_quantizers = len(collect_fake_quantize_layers(compressed_model))
    assert non_weight_quantizers == ref_aq_module_count

    total_quantizations = get_total_quantizations(compressed_model)
    assert total_quantizations == ref_quantizations


def get_shared_conv_test_model():
    rpn_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation=None, padding="same", name="rpn")

    rpn_box_conv = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same", name="rpn-box")

    rpn_class_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding="valid", name="rpn-class")

    inputs = []
    for i in range(1, 3):
        inputs.append(tf.keras.Input(shape=(i * 5, i * 5, 3)))

    rois = []
    for inp in inputs:
        rpn = rpn_conv(inp)
        box_conv = rpn_box_conv(rpn)
        class_conv = rpn_class_conv(rpn)

        _, feature_h, feature_w, num_anchors_per_location = class_conv.get_shape().as_list()

        num_boxes = feature_h * feature_w * num_anchors_per_location
        this_level_scores = tf.reshape(class_conv, [-1, num_boxes, 1])
        this_level_boxes = tf.reshape(box_conv, [-1, num_boxes, 4])

        cat = tf.concat([this_level_scores, this_level_boxes], -1)
        rois.append(cat)

    all_rois = tf.concat(rois, 1)
    out = all_rois[:, :, 3] * all_rois[:, :, 2]
    return tf.keras.Model(inputs=inputs, outputs=out)


@pytest.mark.parametrize("target_device", [t_dev.value for t_dev in HWConfigType])
def test_shared_op_unified_scales(target_device):
    nncf_config = get_basic_quantization_config()
    nncf_config["target_device"] = target_device

    non_weight_quantizers_ref = 8
    if target_device == "NPU":
        non_weight_quantizers_ref = 5

    model = get_shared_conv_test_model()
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config, force_no_init=True)

    non_weight_quantizers = len(collect_fake_quantize_layers(compressed_model))
    assert non_weight_quantizers == non_weight_quantizers_ref

    total_quantizations = get_total_quantizations(compressed_model)
    assert total_quantizations == 8

    input_1 = tf.random.uniform(shape=(1, 5, 5, 3), dtype=tf.float32)
    input_2 = tf.random.uniform(shape=(1, 10, 10, 3), dtype=tf.float32)
    inputs = [input_1, input_2]
    out = compressed_model.predict(inputs)
    assert out.shape == (1, 375)


def get_eltwise_quantizer_linking_test_model(input_shapes):
    inputs = []
    for i, input_shape in enumerate(input_shapes):
        inputs.append(tf.keras.Input(shape=input_shape[1:], name="input_{}".format(i + 1)))

    input_1, input_2 = inputs

    def path(input_1, input_2):
        retval0 = input_1 + input_2
        retval1 = retval0 * input_2
        retval2 = retval0 + retval1
        return retval0, retval1, retval2

    path1_results = path(input_1, input_2)
    path2_results = path(input_1, input_2)

    outputs = tuple(x + y for x, y in zip(path1_results, path2_results))
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_eltwise_unified_scales_for_npu():
    nncf_config = get_basic_quantization_config()
    x_shape = [1, 1, 1, 1]
    y_shape = [1, 1, 1, 1]
    nncf_config["target_device"] = "NPU"

    model = get_eltwise_quantizer_linking_test_model([x_shape, y_shape])
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config, force_no_init=True)

    non_weight_quantizers = len(collect_fake_quantize_layers(compressed_model))
    assert non_weight_quantizers == 2

    total_quantizations = get_total_quantizations(compressed_model)
    assert total_quantizations == 8
