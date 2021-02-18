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
from pathlib import Path
import tensorflow as tf
from addict import Dict

from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss
from beta.nncf.helpers.model_creation import create_compressed_model
from beta.nncf import NNCFConfig
from beta.tests.tensorflow.helpers import get_basic_conv_test_model, \
    create_compressed_model_and_algo_for_test, get_empty_config, get_weight_by_name

CONF = Path(__file__).parent.parent.parent / 'data' / 'configs' / 'sequential_model_cifar10_rb_sparsity.json'


def get_basic_rb_sparse_model(config=CONF, freeze=False):
    model = get_basic_conv_test_model()
    if isinstance(config, Path):
        config = NNCFConfig.from_json(config)
    compress_model, algo = create_compressed_model_and_algo_for_test(model, config)
    if freeze:
       algo.freeze()
    return compress_model, algo, config


class TestSparseModules:
    def test_create_loss__with_defaults(self):
        compr_model, algo, config = get_basic_rb_sparse_model()
        loss = algo.loss
        assert not loss.disabled
        tf.debugging.assert_near(loss.target_sparsity_rate,
                                        config['compression']['params']['multistep_sparsity_levels'][0])
        assert loss.p == 0.05

    @pytest.mark.parametrize(('mask_value', 'ref_loss'),
                             ((None, 4),
                              (0., 0),
                              (0.3, 4),
                              (-0.3, 0)), ids=('default', 'zero', 'positive', 'negative'))
    def test_can_forward_sparse_module__with_frozen_mask(self, mask_value, ref_loss):
        model, algo, conf = get_basic_rb_sparse_model(freeze=True)
        sm = model.layers[1]
        # Set weights
        kernel = get_weight_by_name(sm, 'kernel')
        kernel.assign(tf.ones_like(kernel))
        # Set bias
        bias = get_weight_by_name(sm, 'bias')
        bias.assign(tf.zeros_like(bias))
        if mask_value is not None:
            # Set mask
            mask = get_weight_by_name(sm, 'mask')
            mask.assign(tf.fill(mask.shape, mask_value))
        input_ = tf.ones([4, 4, 4, 1])
        assert tf.reduce_all(model(input_) == ref_loss)

    @pytest.mark.parametrize(('frozen', 'raising'), ((True, True), (False, False)),
                             ids=('frozen', 'not_frozen'))
    def test_calc_loss(self, frozen, raising):
        model, algo, conf = get_basic_rb_sparse_model(freeze=frozen)
        rb_weight = model.layers[1].get_op_by_name('rb_sparsity_mask_apply')
        assert rb_weight.trainable is not frozen
        loss = SparseLoss(algo.loss._sparse_layers)
        try:
            assert loss() == 0
        except ZeroDivisionError:
            pytest.fail("Division by zero")
        except AssertionError:
            if not raising:
                pytest.fail("Exception is not expected")


    @pytest.mark.parametrize('frozen', (None, False, True), ids=('default', 'sparsify', 'frozen'))
    class TestWithSparsify:
        def test_can_freeze_mask(self, frozen):
            model, algo, conf = get_basic_rb_sparse_model(freeze=frozen)
            rb_weight = model.layers[1].get_op_by_name('rb_sparsity_mask_apply')
            if frozen is None:
                frozen = False
            assert rb_weight.trainable is not frozen
            trainable = get_weight_by_name(model.layers[1], 'trainable')
            val = tf.constant(int(not frozen), shape=(), dtype=tf.int8)
            assert trainable == val

        def test_disable_loss(self, frozen):
            model, algo, conf = get_basic_rb_sparse_model(freeze=frozen)
            rb_weight = model.layers[1].get_op_by_name('rb_sparsity_mask_apply')
            assert rb_weight.trainable is (True if frozen is None else not frozen)
            loss = algo.loss
            loss.disable()
            assert not rb_weight.trainable

    @pytest.mark.parametrize(('target', 'expected_rate'),
                             ((None, 0),
                              (0, 1),
                              (0.5, 0.5),
                              (1, 0),
                              (1.5, None),
                              (-0.5, None)),
                             ids=('default', 'min', 'middle', 'max', 'more_than_max', 'less_then_min'))
    def test_get_target_sparsity_rate(self, target, expected_rate):
        config = get_empty_config()
        config['compression'] = Dict({'algorithm': 'rb_sparsity', 'params': {'schedule': 'exponential'}})
        model, algo, conf = get_basic_rb_sparse_model(config=config)
        loss = algo.loss
        if target is not None:
            loss.target = target
        actual_rate = None
        try:
            actual_rate = loss.target_sparsity_rate
            if expected_rate is None:
                pytest.fail("Exception should be raised")
        except IndexError:
            if expected_rate is not None:
                pytest.fail("Exception is not expected")
        if expected_rate is not None:
            assert actual_rate == expected_rate
