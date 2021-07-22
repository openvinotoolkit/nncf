"""
 Copyright (c) 2020 Intel Corporation
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
import random
import contextlib

import pytest
import numpy as np
import tensorflow as tf

from nncf import NNCFConfig
from nncf.tensorflow import create_compression_callbacks
from nncf.tensorflow import register_default_init_args
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.utils import get_basic_quantization_config


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_global_determinism(seed, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_random_seed(seed=seed)
    if fast_n_close:
        return

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def get_simple_conv_regression_model(img_size=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3),
                               activation='relu', input_shape=(img_size, img_size, 1)),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model


def get_basic_magnitude_sparsity_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 4, 4, 1]
    config = NNCFConfig({
        "model": "basic_sparse_conv",
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression":
            {
                "algorithm": "magnitude_sparsity",
                "sparsity_init": 0.3,
                "params": {}
            }
    })
    return config


def get_const_target_mock_regression_dataset(num_samples=20, img_size=10, target_value=20.0):
    class SingleBatchGenerator:
        def __init__(self, X):
            self.X = X

        def __call__(self):
            for i in range(len(self.X)):
                xi = np.expand_dims(self.X[i], axis=0)
                yield xi, [target_value, ]

    X = [np.random.uniform(0, 255, size=(img_size, img_size, 1)).astype(np.uint8)
         for _ in range(num_samples)]
    gen = SingleBatchGenerator(X)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((1, img_size, img_size, 1), (1,)),
    )
    return dataset


@pytest.mark.parametrize(
    ('max_accuracy_degradation',
     'final_compression_rate',
     'reference_final_metric',
     'should_raise_runtime_error'),
    (
            (30.0, 0.846153, 0.1419714, False),
            (1.0, 0.0, 0.0, True),
    )
)
def test_adaptive_compression_training_loop(max_accuracy_degradation, final_compression_rate,
                                            reference_final_metric, should_raise_runtime_error,
                                            initial_training_phase_epochs=5, patience_epochs=3,
                                            uncompressed_model_accuracy=0.2, steps_per_epoch=20,
                                            img_size=10):
    set_random_seed(42)
    model = get_simple_conv_regression_model(img_size)
    dataset = get_const_target_mock_regression_dataset(img_size=img_size,
                                                       num_samples=steps_per_epoch)
    config = get_basic_magnitude_sparsity_config(input_sample_size=[1, img_size, img_size, 1])

    acc_aware_config = {
        'maximal_accuracy_degradation': max_accuracy_degradation,
        'initial_training_phase_epochs': initial_training_phase_epochs,
        'patience_epochs': patience_epochs,
    }
    config['compression']['accuracy_aware_training'] = acc_aware_config

    compress_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    compress_model.add_loss(compression_ctrl.loss)

    def inverse_loss(y_true, y_pred):
        return 1 / (1 + (y_true - y_pred) ** 2)

    compress_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=inverse_loss)

    result_dict_to_val_metric_fn = lambda results: results['inverse_loss']

    exec_ctx = pytest.raises(RuntimeError) if should_raise_runtime_error \
        else contextlib.suppress()
    with exec_ctx as execinfo:
        compress_model.accuracy_aware_fit(dataset,
                                          compression_ctrl,
                                          nncf_config=config,
                                          callbacks=compression_callbacks,
                                          initial_epoch=0,
                                          steps_per_epoch=steps_per_epoch,
                                          uncompressed_model_accuracy=uncompressed_model_accuracy,
                                          result_dict_to_val_metric_fn=result_dict_to_val_metric_fn)
        validation_metrics = compress_model.evaluate(dataset, return_dict=True)

        assert result_dict_to_val_metric_fn(validation_metrics) == pytest.approx(reference_final_metric, 1e-4)
        assert compression_ctrl.compression_rate == pytest.approx(final_compression_rate, 1e-3)

    if should_raise_runtime_error:
        assert str(execinfo.value) == 'Cannot produce a compressed model with a ' \
                                      'specified minimal tolerable accuracy'


@pytest.mark.parametrize(
    ('max_accuracy_degradation',
     'reference_final_metric'),
    (
            (30.0, 0.160190),
            (1.0, 0.202863),
    )
)
def test_early_stopping_compression_training_loop(max_accuracy_degradation,
                                                  reference_final_metric,
                                                  maximal_total_epochs=100, uncompressed_model_accuracy=0.2,
                                                  steps_per_epoch=20, img_size=10):
    set_global_determinism(42)
    model = get_simple_conv_regression_model(img_size)
    dataset = get_const_target_mock_regression_dataset(img_size=img_size,
                                                       num_samples=steps_per_epoch)

    config = get_basic_quantization_config(img_size)
    acc_aware_config = {
        "maximal_accuracy_degradation": max_accuracy_degradation,
        "maximal_total_epochs": maximal_total_epochs,
    }
    config['compression']['accuracy_aware_training'] = acc_aware_config
    config = register_default_init_args(config, dataset, batch_size=1)
    compress_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    compress_model.add_loss(compression_ctrl.loss)

    def inverse_loss(y_true, y_pred):
        return 1 / (1 + (y_true - y_pred) ** 2)

    compress_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=inverse_loss)

    result_dict_to_val_metric_fn = lambda results: results['inverse_loss']

    compress_model.accuracy_aware_fit(dataset,
                                      compression_ctrl,
                                      nncf_config=config,
                                      callbacks=compression_callbacks,
                                      initial_epoch=0,
                                      steps_per_epoch=steps_per_epoch,
                                      uncompressed_model_accuracy=uncompressed_model_accuracy,
                                      result_dict_to_val_metric_fn=result_dict_to_val_metric_fn)
    validation_metrics = compress_model.evaluate(dataset, return_dict=True)

    assert result_dict_to_val_metric_fn(validation_metrics) == pytest.approx(reference_final_metric, 1e-4)
