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
from pytest import approx

from beta.nncf import NNCFConfig
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.sparsity.schedulers import PolynomialSparseScheduler
from beta.nncf.tensorflow.sparsity.rb.algorithm import RBSparsityController
from beta.nncf.tensorflow.sparsity.rb.operation import OP_NAME
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss
from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from beta.nncf.tensorflow.sparsity.rb.functions import logit
from beta.tests.tensorflow.helpers import get_basic_conv_test_model, \
    create_compressed_model_and_algo_for_test, get_weight_by_name, get_basic_two_conv_test_model


def get_basic_sparsity_config(model_size=4, input_sample_size=None,
                              sparsity_init=0.02, sparsity_target=0.5, sparsity_target_epoch=2,
                              sparsity_freeze_epoch=3):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]

    config = NNCFConfig()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression":
            {
                "algorithm": "rb_sparsity",
                "params":
                    {
                        "schedule": "polynomial",
                        "sparsity_init": sparsity_init,
                        "sparsity_target": sparsity_target,
                        "sparsity_target_epoch": sparsity_target_epoch,
                        "sparsity_freeze_epoch": sparsity_freeze_epoch
                    },
            }
    })
    return config


def test_can_load_sparse_algo__with_defaults():
    model = get_basic_two_conv_test_model()
    config = get_basic_sparsity_config(sparsity_init=0.1)
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, RBSparsityController)
    assert compression_ctrl.get_sparsity_init() == approx(0.1)
    assert compression_ctrl.loss.target_sparsity_rate == approx(0.1)

    conv_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    wrappers = [layer for layer in sparse_model.layers if isinstance(layer, NNCFWrapper)]
    correct_wrappers = [wrapper for wrapper in wrappers if wrapper.layer.name in conv_names]

    assert len(conv_names) == len(wrappers)
    assert len(conv_names) == len(correct_wrappers)

    for wrapper in wrappers:
        mask = get_weight_by_name(wrapper, 'mask')
        op = wrapper.get_op_by_name(OP_NAME)
        ref_mask = tf.fill(mask.shape, logit(0.99))

        tf.assert_equal(mask, ref_mask)
        assert isinstance(op, RBSparsifyingWeight)


def test_can_set_sparse_layers_to_loss():
    model = get_basic_conv_test_model()
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)
    # pylint: disable=protected-access
    for layer in loss._sparse_layers:
        assert isinstance(layer, NNCFWrapper)
        assert isinstance(layer.get_op_by_name(OP_NAME), RBSparsifyingWeight)


def test_sparse_algo_does_not_replace_not_conv_layer():
    x = tf.keras.layers.Input((10, 10, 3))
    y = tf.keras.layers.Conv2D(1, 1)(x)
    y = tf.keras.layers.BatchNormalization()(y)

    model = tf.keras.Model(inputs=x, outputs=y)
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, RBSparsityController)
    # pylint: disable=protected-access
    sparse_layers = compression_ctrl.loss._sparse_layers
    assert len(sparse_layers) == 1
    assert isinstance(sparse_layers[0].get_op_by_name(OP_NAME), RBSparsifyingWeight)


def test_can_create_sparse_loss_and_scheduler():
    model = get_basic_conv_test_model()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    scheduler = compression_ctrl.scheduler
    scheduler.epoch_step()
    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)
    assert not loss.disabled
    assert loss.target_sparsity_rate == approx(0.02)
    assert loss.p == approx(0.05)

    assert isinstance(scheduler, PolynomialSparseScheduler)
    assert scheduler.current_sparsity_level == approx(0.02)
    assert scheduler.sparsity_target == approx(0.5)
    assert scheduler.sparsity_target_epoch == 2
    assert scheduler.sparsity_freeze_epoch == 3


def test_sparse_algo_can_collect_sparse_layers():
    model = get_basic_two_conv_test_model()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    # pylint: disable=protected-access
    assert len(compression_ctrl.loss._sparse_layers) == 2


def test_scheduler_can_do_epoch_step__with_rb_algo():
    config = NNCFConfig()
    config['input_info'] = [{"sample_size": [1, 4, 4, 1]}]
    config['compression'] = {
        'algorithm': 'rb_sparsity',
        'sparsity_init': 0.2,
        "params": {
            'schedule': 'polynomial',
            'power': 1,
            'sparsity_target_epoch': 3,
            'sparsity_target': 0.6,
            'sparsity_freeze_epoch': 4
        }
    }

    _, compression_ctrl = create_compressed_model_and_algo_for_test(get_basic_conv_test_model(), config)
    scheduler = compression_ctrl.scheduler
    loss = compression_ctrl.loss

    assert pytest.approx(loss.target_sparsity_rate) == 0.2
    assert not loss.disabled

    # pylint: disable=protected-access
    for wrapper in loss._sparse_layers:
        assert tf.equal(wrapper.ops_weights[OP_NAME]['trainable'], tf.constant(1, dtype=tf.int8))

    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.2
    assert pytest.approx(loss(), abs=1e-3) == 16
    assert not loss.disabled

    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.4
    assert pytest.approx(loss(), abs=1e-3) == 64
    assert not loss.disabled

    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.6
    assert pytest.approx(loss(), abs=1e-3) == 144
    assert not loss.disabled

    scheduler.epoch_step()
    assert loss.disabled
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.6
    assert loss() == 0

    for wrapper in loss._sparse_layers:
        assert tf.equal(wrapper.ops_weights[OP_NAME]['trainable'], tf.constant(0, dtype=tf.int8))
