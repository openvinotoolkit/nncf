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

from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from nncf.tensorflow.sparsity.rb.functions import st_binary_mask
from nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from tests.tensorflow.sparsity.rb.utils import default_rb_mask_value


def get_RBSParsityWeigth_and_layer(frozen=False):
    op_name = "rb_op_name"
    sw = RBSparsifyingWeight(op_name)
    layer = NNCFWrapper(tf.keras.layers.Conv2D(1, 1))
    layer.registry_weight_operation("kernel", sw)
    layer.build((1,))
    if frozen:
        sw.freeze(layer.ops_weights[op_name])
    return sw, layer, op_name


def test_can_create_sparse_weight__with_defaults():
    sw, layer, op_name = get_RBSParsityWeigth_and_layer()
    weights = layer.ops_weights[op_name]
    trainable_weight = weights["trainable"]
    mask = weights["mask"]
    tf.debugging.assert_near(default_rb_mask_value, mask)
    assert trainable_weight
    assert sw.eps == 1e-6


def test_can_freeze_mask():
    sw, layer, op_name = get_RBSParsityWeigth_and_layer(frozen=False)
    op_weights = layer.ops_weights[op_name]
    assert op_weights["trainable"]
    sw.freeze(op_weights)
    assert not op_weights["trainable"]


@pytest.mark.parametrize("frozen", (True, False), ids=("sparsify", "frozen"))
class TestWithSparsify:
    @pytest.mark.parametrize("is_train", (True, False), ids=("train", "not_train"))
    def test_mask_is_not_updated_on_forward(self, frozen, is_train):
        sw, layer, op_name = get_RBSParsityWeigth_and_layer(frozen=frozen)
        op_weights = layer.ops_weights[op_name]
        mask = op_weights["mask"]

        tf.debugging.assert_near(default_rb_mask_value, mask)
        w = tf.ones(1)
        sw(w, op_weights, tf.constant(is_train, tf.bool))
        tf.debugging.assert_near(default_rb_mask_value, mask)

    @pytest.mark.parametrize(
        ("mask_value", "ref_loss"),
        ((None, 1), (0.0, 0), (0.3, 1), (-0.3, 0)),
        ids=("default", "zero", "positive", "negative"),
    )
    def test_loss_value(self, mask_value, ref_loss, frozen):
        sw, layer, op_name = get_RBSParsityWeigth_and_layer(frozen=frozen)
        op_weights = layer.ops_weights[op_name]
        mask = op_weights["mask"]
        if mask_value is not None:
            mask.assign(tf.fill(mask.shape, mask_value))
        assert sw.loss(op_weights) == ref_loss
        w = tf.ones(1)
        assert apply_mask(w, st_binary_mask(mask)) == ref_loss
        sw.freeze(op_weights)
        assert sw(w, op_weights, tf.constant(True, tf.bool)) == ref_loss
