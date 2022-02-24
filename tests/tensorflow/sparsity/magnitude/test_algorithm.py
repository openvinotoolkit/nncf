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

import numpy as np
import pytest
import tensorflow as tf
from addict import Dict
from pytest import approx

from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.sparsity.magnitude.algorithm import MagnitudeSparsityController
from nncf.tensorflow.sparsity.magnitude.functions import normed_magnitude
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.helpers import get_mock_model
from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.tensorflow.sparsity.magnitude.test_helpers import get_magnitude_test_model
from tests.tensorflow.sparsity.magnitude.test_helpers import ref_mask_1
from tests.tensorflow.sparsity.magnitude.test_helpers import ref_mask_2


def test_can_create_magnitude_sparse_algo__with_defaults():
    model = get_magnitude_test_model()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = \
        {'schedule': 'multistep'}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, MagnitudeSparsityController)
    assert compression_ctrl.scheduler.current_sparsity_level == approx(0.1)

    conv_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    wrappers = [layer for layer in sparse_model.layers if isinstance(layer, NNCFWrapper)]
    correct_wrappers = [wrapper for wrapper in wrappers if wrapper.name in conv_names]

    assert len(conv_names) == len(wrappers)
    assert len(conv_names) == len(correct_wrappers)

    assert compression_ctrl._threshold == approx(0.24, 0.1) # pylint: disable=protected-access
    # pylint: disable=protected-access
    assert isinstance(compression_ctrl._weight_importance_fn, type(normed_magnitude))

    for i, wrapper in enumerate(wrappers):
        ref_mask = tf.ones_like(wrapper.weights[-1]) if i == 0 else ref_mask_2
        mask = list(wrapper.ops_weights.values())[0]['mask']
        op = list(wrapper.weights_attr_ops['kernel'].values())[0]

        tf.assert_equal(mask, ref_mask)
        assert isinstance(op, BinaryMask)


def test_compression_controller_state():
    from nncf.common.compression import BaseControllerStateNames as CtrlStateNames
    model = get_magnitude_test_model()
    config = get_basic_magnitude_sparsity_config()
    algo_name = config['compression']['algorithm']
    config['compression']['params'] = {'schedule': 'multistep'}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    # Test get state
    compression_ctrl.scheduler.load_state({'current_step': 100, 'current_epoch': 5})
    state_content = compression_ctrl.get_state()[algo_name]
    assert state_content[CtrlStateNames.SCHEDULER] == {'current_step': 100, 'current_epoch': 5}

    # Test load state
    new_state = {
        algo_name: {
            CtrlStateNames.SCHEDULER: {'current_step': 500, 'current_epoch': 10},
            CtrlStateNames.LOSS: {},
            CtrlStateNames.COMPRESSION_STAGE: None,
        }
    }
    compression_ctrl.load_state(new_state)
    assert compression_ctrl.scheduler.current_step == 500
    assert compression_ctrl.scheduler.current_epoch == 10
    assert compression_ctrl.get_state() == new_state


@pytest.mark.parametrize(
    ('weight_importance', 'sparsity_level', 'threshold'),
    (
        ('normed_abs', None, 0.219),
        ('abs', None, 9),
        ('normed_abs', 0.5, 0.243),
        ('abs', 0.5, 10),
    )
)
def test_magnitude_sparse_algo_sets_threshold(weight_importance, sparsity_level, threshold):
    model = get_magnitude_test_model()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep',
                                       'weight_importance': weight_importance}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if sparsity_level:
        compression_ctrl.set_sparsity_level(sparsity_level)
    assert compression_ctrl._threshold == pytest.approx(threshold, 0.01) # pylint: disable=protected-access


def test_can_create_magnitude_algo__without_levels():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'multistep_steps': [1]}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(get_mock_model(), config)
    assert compression_ctrl.scheduler.current_sparsity_level == approx(0.1)


def test_can_not_create_magnitude_algo__with_not_matched_steps_and_levels():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'multistep_sparsity_levels': [0.1],
                                       'multistep_steps': [1, 2]}
    with pytest.raises(ValueError):
        _, _ = create_compressed_model_and_algo_for_test(get_mock_model(), config)


def test_magnitude_algo_set_binary_mask_on_forward():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'weight_importance': 'abs'}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(get_magnitude_test_model(), config)
    compression_ctrl.set_sparsity_level(0.3)

    TFTensorListComparator.check_equal(ref_mask_1, sparse_model.layers[1].weights[-1])
    TFTensorListComparator.check_equal(ref_mask_2, sparse_model.layers[2].weights[-1])


def test_magnitude_algo_binary_masks_are_applied():
    input_shape = (1, 5, 5, 1)
    model = get_basic_conv_test_model(input_shape=input_shape[1:])
    config = get_empty_config(input_sample_sizes=input_shape)
    config.update(Dict({'compression': {'algorithm': "magnitude_sparsity"}}))
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    conv = compressed_model.layers[1]
    op_name = list(conv.ops_weights.keys())[0]
    conv.ops_weights[op_name] = {'mask': tf.ones_like(conv.weights[0])}
    input_ = tf.ones(input_shape)
    ref_output_1 = -4 * tf.ones((1, 4, 4, 2))
    output_1 = compressed_model(input_)
    tf.assert_equal(output_1, ref_output_1)

    np_mask = conv.ops_weights[op_name]['mask'].numpy()
    np_mask[0, 1, 0, 0] = 0
    np_mask[1, 0, 0, 1] = 0
    conv.ops_weights[op_name] = {'mask': tf.constant(np_mask)}
    ref_output_2 = - 3 * tf.ones_like(ref_output_1)
    output_2 = compressed_model(input_)
    tf.assert_equal(output_2, ref_output_2)

    np_mask[0, 1, 0, 1] = 0
    conv.ops_weights[op_name] = {'mask': tf.constant(np_mask)}
    ref_output_3 = ref_output_2.numpy()
    ref_output_3[..., 1] = -2 * np.ones_like(ref_output_1[..., 1])
    ref_output_3 = tf.constant(ref_output_3)
    output_3 = compressed_model(input_)
    tf.assert_equal(output_3, ref_output_3)
