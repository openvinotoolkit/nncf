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
import tensorflow_addons as tfa
from addict import Dict

from beta.nncf import NNCFConfig
from beta.nncf.tensorflow.sparsity.rb.functions import binary_mask
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss
from beta.tests.tensorflow.sparsity.rb.utils import default_rb_mask_value
from beta.tests.tensorflow.helpers import get_basic_conv_test_model, get_basic_fc_test_model, \
    create_compressed_model_and_algo_for_test, get_empty_config, get_weight_by_name

CONF = Path(__file__).parent.parent.parent / 'data' / 'configs' / 'sequential_model_cifar10_rb_sparsity.json'

TEST_MODELS = {

    'Dense': lambda: get_basic_fc_test_model(
                                        input_shape=(4, ),
                                        out_shape=10),
    'Conv2D': lambda: get_basic_conv_test_model(
                                        input_shape=(4, 4, 1),
                                        out_channels=2,
                                        kernel_size=2,
                                        weight_init=-1.,
                                        bias_init=-2.,
                                        transpose=False),
    'Conv2DTranspose': lambda: get_basic_conv_test_model(
                                        input_shape=(4, 4, 1),
                                        out_channels=1,
                                        kernel_size=2,
                                        weight_init=-1.,
                                        bias_init=-2.,
                                        transpose=True),
}


def get_basic_rb_sparse_model(model_name, config=CONF, freeze=False):
    model = TEST_MODELS[model_name]()
    if isinstance(config, Path):
        config = NNCFConfig.from_json(config)
    compress_model, algo = create_compressed_model_and_algo_for_test(model, config)
    if freeze:
       algo.freeze()
    return compress_model, algo, config


@pytest.mark.parametrize('model_name',
                         list(TEST_MODELS.keys()), ids=list(TEST_MODELS.keys()))
class TestSparseModules:
    def test_create_loss__with_defaults(self, model_name):
        compr_model, algo, config = get_basic_rb_sparse_model(model_name)
        loss = algo.loss
        assert not loss.disabled
        tf.debugging.assert_near(loss.target_sparsity_rate,
                                        tf.Variable(config['compression']['params']['multistep_sparsity_levels'][0],
                                                    dtype=tf.float64),
                                 rtol=tf.Variable(1e-6, dtype=tf.float64))
        # TODO: too big tolerance due to python float error
        assert loss.p == 0.05

    REF_LOSS_IF_NOT_FROZEN = {
        'Dense': tf.fill((10, ), 4.),
        'Conv2D': tf.fill((3, 3, 1), 4.),
        'Conv2DTranspose': tf.reshape(tf.constant([
                                        [1, 2, 2, 2, 1],
                                        [2, 4, 4, 4, 2],
                                        [2, 4, 4, 4, 2],
                                        [2, 4, 4, 4, 2],
                                        [1, 2, 2, 2, 1]], dtype=tf.float32), (1, 5, 5, 1)),

    }

    @pytest.mark.parametrize(('mask_value', 'zero_mask'),
                             ((None, False),
                              (0., True),
                              (0.3, False),
                              (-0.3, True)), ids=('default', 'zero', 'positive', 'negative'))
    def test_can_forward_sparse_module__with_frozen_mask(self, model_name, mask_value, zero_mask):
        model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=True)
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
        input_ = tf.ones((1, ) + model.input_shape[1:])
        output_ = model(input_)
        if zero_mask:
            assert tf.reduce_all(output_ == 0)
        else:
            assert tf.reduce_all(output_ == self.REF_LOSS_IF_NOT_FROZEN[model_name])

    @pytest.mark.parametrize(('frozen', 'raising'), ((True, True), (False, False)),
                             ids=('frozen', 'not_frozen'))
    def test_calc_loss(self, model_name, frozen, raising):
        model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=frozen)
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


    @pytest.mark.parametrize('frozen', (False, True), ids=('sparsify', 'frozen'))
    class TestWithSparsify:
        def test_can_freeze_mask(self, model_name, frozen):
            model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=frozen)
            rb_weight = model.layers[1].get_op_by_name('rb_sparsity_mask_apply')
            assert rb_weight.trainable is not frozen
            trainable = get_weight_by_name(model.layers[1], 'trainable')
            val = tf.constant(int(not frozen), shape=(), dtype=tf.int8)
            assert trainable == val

        def test_disable_loss(self, model_name, frozen):
            model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=frozen)
            rb_weight = model.layers[1].get_op_by_name('rb_sparsity_mask_apply')
            assert rb_weight.trainable is not frozen
            loss = algo.loss
            loss.disable()
            assert not rb_weight.trainable

        def test_check_gradient_existing(self, model_name, frozen):
            model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=frozen)

            algo.loss.set_target_sparsity_loss(1.0)
            dataset_len = (1, )
            dummy_x = tf.random.normal(dataset_len + model.input_shape[1:])
            dummy_y = tf.random.normal(dataset_len + model.output_shape[1:])

            loss_ce = tf.keras.losses.CategoricalCrossentropy()

            with tf.GradientTape() as tape:
                output = model(dummy_x)
                loss = loss_ce(dummy_y, output) + algo.loss()

            grads = tape.gradient(loss, model.trainable_weights)
            grads_weights_paris = list(zip(grads, model.trainable_weights))
            assert all([g is not None for g, w in grads_weights_paris if 'mask' not in w.name])
            assert all([g is None if frozen else not None for g, w in grads_weights_paris if 'mask' in w.name])

        def test_masks_gradients(self, model_name, frozen):
            model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=frozen)

            algo.loss.set_target_sparsity_loss(1.0)

            optimizer_fn = tf.keras.optimizers.SGD(10)
            for step in range(1):
                with tf.GradientTape() as tape:
                    loss = algo.loss()
                grads = tape.gradient(loss, model.trainable_weights)
                if frozen:
                    assert all([x is None for x in grads])
                    continue
                # Keep only masks
                grad_pairs = [(grad, weight) for grad, weight in zip(grads, model.trainable_weights)
                              if 'mask' in weight.name]
                optimizer_fn.apply_gradients(grad_pairs)

            for layer in algo.loss._sparse_layers:
                for weight in layer.weights:
                    if 'mask' in weight.name:
                        if not frozen:
                            assert tf.reduce_all(binary_mask(weight) == 0.)
                        else:
                            tf.debugging.assert_near(weight, tf.fill(weight.shape, default_rb_mask_value))

        def test_keras_train_loop(self, model_name, frozen):
            model, algo, conf = get_basic_rb_sparse_model(model_name, freeze=frozen)

            model.add_loss(algo.loss)

            algo.loss.set_target_sparsity_loss(1.0)
            dataset_len = (1, )
            dummy_x = tf.random.normal(dataset_len + model.input_shape[1:])
            dummy_y = tf.random.normal(dataset_len + model.output_shape[1:])

            model.compile(optimizer=tf.keras.optimizers.SGD(10),
                          loss=[tf.keras.losses.CategoricalCrossentropy()],
                          metrics=[tfa.metrics.MeanMetricWrapper(algo.loss, name='rb_loss')])
            model.fit(x=dummy_x,
                      y=dummy_y,
                      batch_size=1,
                      epochs=1)

            for layer in algo.loss._sparse_layers:
                for weight in layer.weights:
                    if 'mask' in weight.name:
                        if not frozen:
                            assert tf.reduce_all(binary_mask(weight) == 0.)
                        else:
                            tf.debugging.assert_near(weight, tf.fill(weight.shape, default_rb_mask_value))


    @pytest.mark.parametrize(('target', 'expected_rate'),
                             ((None, 0),
                              (0, 1),
                              (0.5, 0.5),
                              (1, 0),
                              (1.5, None),
                              (-0.5, None)),
                             ids=('default', 'min', 'middle', 'max', 'more_than_max', 'less_then_min'))
    def test_get_target_sparsity_rate(self, model_name, target, expected_rate):
        config = get_empty_config()
        config['compression'] = Dict({'algorithm': 'rb_sparsity', 'params': {'schedule': 'exponential'}})
        model, algo, conf = get_basic_rb_sparse_model(model_name, config=config)
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
