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

import pytest
import tensorflow as tf

from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight, OP_NAME
from beta.nncf.tensorflow.sparsity.rb.functions import st_binary_mask
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from beta.tests.tensorflow.helpers import get_weight_by_name
from beta.tests.tensorflow.sparsity.rb.utils import default_rb_mask_value


def get_RBSParsityWeigth_and_layer(frozen=False):
    sw = RBSparsifyingWeight()
    layer = NNCFWrapper(tf.keras.layers.Conv2D(1, 1))
    layer.registry_weight_operation('kernel', sw)
    layer.build((1, ))
    sw.build((1, ), InputType.WEIGHTS, 'dummy_layer_name', layer)
    if frozen:
        sw.freeze(layer.ops_weights[OP_NAME])
    return sw, layer


def test_can_create_sparse_weight__with_defaults():
    sw, layer = get_RBSParsityWeigth_and_layer()
    trainable_weight = get_weight_by_name(layer, 'trainable')
    mask = get_weight_by_name(layer, 'mask')
    assert sw.trainable
    tf.debugging.assert_near(default_rb_mask_value, mask)
    assert trainable_weight == tf.constant(1, dtype=tf.int8)
    assert sw.eps == 1e-6


def test_can_freeze_mask():
    sw, layer = get_RBSParsityWeigth_and_layer(frozen=False)
    trainable_weight = get_weight_by_name(layer, 'trainable')
    assert sw.trainable
    assert trainable_weight == tf.constant(1, dtype=tf.int8)
    sw.freeze(layer.ops_weights[OP_NAME])
    assert not sw.trainable
    assert trainable_weight == tf.constant(0, dtype=tf.int8)


@pytest.mark.parametrize('frozen', (True, False), ids=('sparsify', 'frozen'))
class TestWithSparsify:
    @pytest.mark.parametrize('is_train', (True, False), ids=('train', 'not_train'))
    def test_mask_is_not_updated_on_forward(self, frozen, is_train):
        sw, layer = get_RBSParsityWeigth_and_layer(frozen=frozen)
        mask = get_weight_by_name(layer, 'mask')
        tf.debugging.assert_near(default_rb_mask_value, mask)
        op_weights = layer.ops_weights[OP_NAME]
        w = tf.ones(1)
        sw(w, op_weights, None)
        tf.debugging.assert_near(default_rb_mask_value, mask)

    @pytest.mark.parametrize(('mask_value', 'ref_loss'),
                             ((None, 1),
                              (0.0, 0),
                              (0.3, 1),
                              (-0.3, 0)), ids=('default', 'zero', 'positive', 'negative'))
    def test_loss_value(self, mask_value, ref_loss, frozen):
        sw, layer = get_RBSParsityWeigth_and_layer(frozen=frozen)
        mask = get_weight_by_name(layer, 'mask')
        op_weights = layer.ops_weights[OP_NAME]
        if mask_value is not None:
            mask.assign(tf.fill(mask.shape, mask_value))
        assert sw.loss(mask) == ref_loss
        w = tf.ones(1)
        assert apply_mask(w, st_binary_mask(mask)) == ref_loss
        sw.freeze(op_weights)
        assert sw(w, op_weights, None) == ref_loss
