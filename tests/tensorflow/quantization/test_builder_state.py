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
import os

import tensorflow as tf

from examples.tensorflow.classification.main import load_checkpoint
from examples.tensorflow.classification.main import load_compression_state
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.tensorflow import create_compression_callbacks
from nncf.tensorflow import register_default_init_args
from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback
from nncf.tensorflow.quantization.algorithm import QuantizationBuilder
from nncf.tensorflow.quantization.algorithm import QuantizationController
from nncf.tensorflow.utils.state import TFCompressionState
from tests.common.serialization import check_serialization
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.quantization.test_algorithm_quantization import check_default_qspecs
from tests.tensorflow.quantization.test_algorithm_quantization import check_specs_for_disabled_saturation_fix
from tests.tensorflow.quantization.utils import get_basic_quantization_config
from tests.tensorflow.test_bn_adaptation import get_dataset_for_test
from tests.tensorflow.test_callbacks import REF_CKPT_DIR


def test_quantization_configs__on_resume_with_compression_state(tmp_path, mocker):
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    init_spy = mocker.spy(QuantizationBuilder, 'initialize')
    gen_setup_spy = mocker.spy(QuantizationBuilder, '_get_quantizer_setup')
    dataset = get_dataset_for_test(shape=[4, 4, 1])
    config = register_default_init_args(config, dataset, 10)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, QuantizationController)
    init_spy.assert_called()
    gen_setup_spy.assert_called()
    saved_quantizer_setup = gen_setup_spy.spy_return
    check_serialization(saved_quantizer_setup, SingleConfigQuantizerSetup.equivalent_to)

    compression_state_to_load = _save_and_load_compression_state(compression_ctrl, tmp_path)

    init_spy.reset_mock()
    gen_setup_spy.reset_mock()

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config,
                                                                                    compression_state_to_load)
    assert isinstance(compression_ctrl, QuantizationController)

    init_spy.assert_not_called()
    gen_setup_spy.assert_not_called()
    check_default_qspecs(compression_model)

    builder = QuantizationBuilder(config)
    builder.load_state(compression_state_to_load['builder_state'])
    # pylint:disable=protected-access
    loaded_quantizer_setup = builder._quantizer_setup
    assert loaded_quantizer_setup.equivalent_to(saved_quantizer_setup)


def _save_and_load_compression_state(compression_ctrl, tmp_path):
    checkpoint_path = tmp_path / 'compression_state'
    checkpoint_to_save = tf.train.Checkpoint(compression_state=TFCompressionState(compression_ctrl))
    checkpoint_to_save.save(checkpoint_path)

    compression_state = load_compression_state(str(checkpoint_path.parent))

    return compression_state


def test_quantization_configs__disable_saturation_fix_and_resume_from_compression_state(tmp_path):
    model = get_basic_conv_test_model()

    config = get_basic_quantization_config()
    config['compression'].update({
        'disable_saturation_fix': True
    })
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)

    compression_state_to_load = _save_and_load_compression_state(compression_ctrl, tmp_path)

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config,
                                                                                    compression_state_to_load)
    assert isinstance(compression_ctrl, QuantizationController)
    check_specs_for_disabled_saturation_fix(compression_model)


def test_checkpoint_callback_make_checkpoints(mocker, tmp_path):
    save_freq = 2
    config = get_basic_quantization_config()
    gen_setup_spy = mocker.spy(QuantizationBuilder, '_get_quantizer_setup')

    model, compression_ctrl = create_compressed_model_and_algo_for_test(get_basic_conv_test_model(),
                                                                        config, force_no_init=True)
    assert isinstance(compression_ctrl, QuantizationController)

    quantizer_setup = gen_setup_spy.spy_return
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    dataset_len = 8

    dummy_x = tf.random.normal((dataset_len,) + model.input_shape[1:])
    dummy_y = tf.random.normal((dataset_len,) + model.output_shape[1:])

    model.compile(loss=tf.losses.CategoricalCrossentropy())

    ckpt_path = tmp_path / 'checkpoint'
    ckpt = tf.train.Checkpoint(model=model, compression_state=TFCompressionState(compression_ctrl))
    model.fit(dummy_x, dummy_y,
              epochs=5,
              batch_size=2,
              callbacks=[CheckpointManagerCallback(ckpt, str(ckpt_path), save_freq), *compression_callbacks])

    assert sorted(os.listdir(ckpt_path)) == REF_CKPT_DIR[save_freq]

    new_compression_state = load_compression_state(ckpt_path)

    new_model, new_compression_ctrl = create_compressed_model_and_algo_for_test(get_basic_conv_test_model(),
                                                                                config, new_compression_state)
    new_model.compile(loss=tf.losses.CategoricalCrossentropy())
    new_ckpt = tf.train.Checkpoint(model=new_model, compression_state=TFCompressionState(new_compression_ctrl))
    load_checkpoint(new_ckpt, ckpt_path)

    builder = QuantizationBuilder(config)
    builder.load_state(new_compression_state['builder_state'])
    # pylint:disable=protected-access
    new_quantizer_setup = builder._quantizer_setup

    assert quantizer_setup.equivalent_to(new_quantizer_setup)
    assert new_compression_ctrl.get_state() == compression_ctrl.get_state()
    assert tf.reduce_all([tf.reduce_all(w_new == w) for w_new, w in zip(new_model.weights, model.weights)])
