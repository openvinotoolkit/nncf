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

import tensorflow as tf

from nncf import NNCFConfig
from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.helpers import create_conv

sub_tensor = tf.constant([[[[1.0, 0.0], [0.0, 1.0]]]])
ref_mask_1 = tf.concat((sub_tensor, sub_tensor), 0)
# OIHW -> HWIO
ref_mask_1 = tf.transpose(ref_mask_1, (2, 3, 1, 0))

sub_tensor = tf.constant([[[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]])
ref_mask_2 = tf.concat((sub_tensor, sub_tensor), 1)
# OIHW -> HWIO
ref_mask_2 = tf.transpose(ref_mask_2, (2, 3, 1, 0))


def get_magnitude_test_model(input_shape=(4, 4, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    x = create_conv(1, 2, 2, 9.0, -2.0)(inputs)
    outputs = create_conv(2, 1, 3, -10.0, 0.0)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_magnitude_model_has_expected_params():
    model = get_magnitude_test_model()
    act_weights_1 = model.layers[1].kernel
    act_weights_2 = model.layers[2].kernel
    act_bias_1 = model.layers[1].bias
    act_bias_2 = model.layers[2].bias

    sub_tensor_ = tf.constant([[[[10.0, 9.0], [9.0, 10.0]]]])
    ref_weights_1 = tf.concat((sub_tensor_, sub_tensor_), 0)
    # OIHW -> HWIO
    ref_weights_1 = tf.transpose(ref_weights_1, (2, 3, 1, 0))

    sub_tensor_ = tf.constant([[[[-9.0, -10.0, -10.0], [-10.0, -9.0, -10.0], [-10.0, -10.0, -9.0]]]])
    ref_weights_2 = tf.concat((sub_tensor_, sub_tensor_), 1)
    # OIHW -> HWIO
    ref_weights_2 = tf.transpose(ref_weights_2, (2, 3, 1, 0))

    TFTensorListComparator.check_equal(act_weights_1, ref_weights_1)
    TFTensorListComparator.check_equal(act_weights_2, ref_weights_2)

    TFTensorListComparator.check_equal(act_bias_1, tf.constant([-2.0, -2]))
    TFTensorListComparator.check_equal(act_bias_2, tf.constant([0]))


def get_basic_magnitude_sparsity_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 4, 4, 1]
    config = NNCFConfig(
        {
            "model": "basic_sparse_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"algorithm": "magnitude_sparsity", "params": {}},
        }
    )
    return config


def get_basic_filter_pruning_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 4, 4, 1]
    config = NNCFConfig(
        {
            "model": "basic_prune_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"algorithm": "filter_pruning", "params": {}},
        }
    )
    return config


def get_basic_sparsity_config(input_sample_size=None, algo="magnitude_sparsity"):
    if input_sample_size is None:
        input_sample_size = [1, 4, 4, 1]
    config = NNCFConfig(
        {
            "model": "basic_sparse_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"algorithm": algo, "params": {}},
        }
    )
    return config
