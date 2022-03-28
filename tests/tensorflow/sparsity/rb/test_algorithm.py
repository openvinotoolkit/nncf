"""
 Copyright (c) 2022 Intel Corporation
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

from nncf import NNCFConfig
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.common.sparsity.schedulers import PolynomialSparsityScheduler
from nncf.tensorflow.sparsity.rb.algorithm import RBSparsityController
from nncf.tensorflow.sparsity.rb.loss import SparseLoss
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from nncf.tensorflow.sparsity.rb.functions import logit
from tests.tensorflow.helpers import get_op_by_cls
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_two_conv_test_model
from tests.tensorflow.helpers import get_basic_conv_test_model


def get_basic_sparsity_config(model_size=4, input_sample_size=None,
                              sparsity_init=0.02, sparsity_target=0.5, sparsity_target_epoch=2,
                              sparsity_freeze_epoch=3):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]

    config = NNCFConfig()
    config.update({
        'model': 'basic_sparse_conv',
        'model_size': model_size,
        'input_info':
            {
                'sample_size': input_sample_size,
            },
        'compression':
            {
                'algorithm': 'rb_sparsity',
                'sparsity_init': sparsity_init,
                'params':
                    {
                        'schedule': 'polynomial',
                        'sparsity_target': sparsity_target,
                        'sparsity_target_epoch': sparsity_target_epoch,
                        'sparsity_freeze_epoch': sparsity_freeze_epoch
                    },
            }
    })
    return config


def test_can_load_sparse_algo__with_defaults():
    model = get_basic_two_conv_test_model()
    config = get_basic_sparsity_config(sparsity_init=0.1)
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, RBSparsityController)
    assert compression_ctrl.scheduler.initial_level == approx(0.1)

    conv_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    wrappers = [layer for layer in sparse_model.layers if isinstance(layer, NNCFWrapper)]
    correct_wrappers = [wrapper for wrapper in wrappers if wrapper.name in conv_names]

    assert len(conv_names) == len(wrappers)
    assert len(conv_names) == len(correct_wrappers)

    for wrapper in wrappers:
        op = get_op_by_cls(wrapper, RBSparsifyingWeight)
        mask = wrapper.get_operation_weights(op.name)['mask']
        ref_mask = tf.fill(mask.shape, logit(0.99))

        tf.assert_equal(mask, ref_mask)


def test_compression_controller_state():
    from nncf.common.compression import BaseControllerStateNames as CtrlStateNames
    model = get_basic_two_conv_test_model()
    config = get_basic_sparsity_config()
    algo_name = config['compression']['algorithm']
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    # Test get state
    compression_ctrl.scheduler.load_state({'current_step': 100, 'current_epoch': 5})
    compression_ctrl.set_sparsity_level(0.5)
    compression_ctrl.freeze()
    assert compression_ctrl.get_state() == {
        algo_name: {
            CtrlStateNames.SCHEDULER: {'current_step': 100, 'current_epoch': 5},
            CtrlStateNames.LOSS: {'target': 0.5, 'disabled': True, 'p': 0.05},
            CtrlStateNames.COMPRESSION_STAGE: None
        }
    }
    # Test load state
    new_state = {
        CtrlStateNames.SCHEDULER: {'current_step': 5000, 'current_epoch': 10},
        CtrlStateNames.LOSS: {'target': 0.9, 'disabled': False, 'p': 0.5},
        CtrlStateNames.COMPRESSION_STAGE: None
    }
    compression_ctrl.load_state({algo_name: new_state})
    assert tf.equal(compression_ctrl.loss.target, tf.constant(new_state[CtrlStateNames.LOSS]['target']))
    assert compression_ctrl.loss.disabled == new_state[CtrlStateNames.LOSS]['disabled']
    assert compression_ctrl.loss.p == pytest.approx(new_state[CtrlStateNames.LOSS]['p'])

    new_real_state = compression_ctrl.get_state()[algo_name]
    assert new_real_state[CtrlStateNames.SCHEDULER] == new_state[CtrlStateNames.SCHEDULER]
    assert new_real_state[CtrlStateNames.LOSS]['target'] == pytest.approx(new_state[CtrlStateNames.LOSS]['target'])
    assert new_real_state[CtrlStateNames.LOSS]['disabled'] == new_state[CtrlStateNames.LOSS]['disabled']
    assert new_real_state[CtrlStateNames.LOSS]['p'] == pytest.approx(new_state[CtrlStateNames.LOSS]['p'])


def test_can_set_sparse_layers_to_loss():
    model = get_basic_conv_test_model()
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)
    # pylint: disable=protected-access
    for op, _ in loss._target_ops:
        assert isinstance(op, RBSparsifyingWeight)


def test_loss_has_correct_ops():
    inp = tf.keras.layers.Input((10, 10, 3))
    y = inp
    for _ in range(3):
        y = tf.keras.layers.Conv2D(1, 1)(y)
    y = tf.keras.layers.BatchNormalization()(y)

    model = tf.keras.Model(inputs=inp, outputs=y)
    config = get_basic_sparsity_config()
    compress_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    wrappers = collect_wrapped_layers(compress_model)
    # pylint: disable=protected-access
    target_ops = {op[0].name: op for op in compression_ctrl.loss._target_ops}
    for wrapper in wrappers:
        for ops in wrapper.weights_attr_ops.values():
            # find corresponding op in target_ops
            op = list(ops.values())[0]
            assert op.name in target_ops
            target_op = target_ops[op.name]
            weights = wrapper.get_operation_weights(op.name)
            assert op is target_op[0]
            assert weights is target_op[1]



def test_sparse_algo_does_not_replace_not_conv_layer():
    x = tf.keras.layers.Input((10, 10, 3))
    y = tf.keras.layers.Conv2D(1, 1)(x)
    y = tf.keras.layers.BatchNormalization()(y)

    model = tf.keras.Model(inputs=x, outputs=y)
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, RBSparsityController)
    # pylint: disable=protected-access
    target_ops = compression_ctrl.loss._target_ops
    assert len(target_ops) == 1
    assert isinstance(target_ops[0][0], RBSparsifyingWeight)


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

    assert isinstance(scheduler, PolynomialSparsityScheduler)
    assert scheduler.current_sparsity_level == approx(0.02)
    assert scheduler.target_level == approx(0.5)
    assert scheduler.target_epoch == 2
    assert scheduler.freeze_epoch == 3


def test_sparse_algo_can_collect_sparse_ops():
    model = get_basic_two_conv_test_model()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    # pylint: disable=protected-access
    assert len(compression_ctrl.loss._target_ops) == 2


def test_scheduler_can_do_epoch_step__with_rb_algo():
    config = NNCFConfig()
    config['input_info'] = [{"sample_size": [1, 4, 4, 1]}]
    config['compression'] = {
        'algorithm': 'rb_sparsity',
        'sparsity_init': 0.2,
        "params": {
            'schedule': 'polynomial',
            'power': 1,
            'sparsity_target_epoch': 2,
            'sparsity_target': 0.6,
            'sparsity_freeze_epoch': 3
        }
    }

    _, compression_ctrl = create_compressed_model_and_algo_for_test(get_basic_conv_test_model(), config)
    scheduler = compression_ctrl.scheduler
    loss = compression_ctrl.loss

    assert not loss.disabled

    # pylint: disable=protected-access
    for op, op_weights in loss._target_ops:
        assert op.get_trainable_weight(op_weights)

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

    for op, op_weights in loss._target_ops:
        assert not op.get_trainable_weight(op_weights)
