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

import random
from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

from nncf import NNCFConfig
from nncf.tensorflow import create_compression_callbacks
from nncf.tensorflow import register_default_init_args
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.utils import get_basic_quantization_config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_simple_conv_regression_model(img_size=10):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=(3, 3), activation="relu", input_shape=(img_size, img_size, 1)
            ),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ]
    )
    return model


def get_basic_magnitude_sparsity_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 4, 4, 1]
    config = NNCFConfig(
        {
            "model": "basic_sparse_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"algorithm": "magnitude_sparsity", "sparsity_init": 0.3, "params": {}},
        }
    )
    return config


def get_const_target_mock_regression_dataset(num_samples=20, img_size=10, target_value=20.0):
    class SingleBatchGenerator:
        def __init__(self, X):
            self.X = X

        def __call__(self):
            for i, _ in enumerate(self.X):
                xi = np.expand_dims(self.X[i], axis=0)
                yield xi, [
                    target_value,
                ]

    X = [np.random.uniform(0, 255, size=(img_size, img_size, 1)).astype(np.uint8) for _ in range(num_samples)]
    gen = SingleBatchGenerator(X)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((1, img_size, img_size, 1), (1,)),
    )
    return dataset


class MockResultFunctor:
    """Returns a mock value first, then parses the real inverse loss from the model's output dict."""

    def __init__(self, mock_retval: float):
        self._call_count = 0
        self._mock_retval = mock_retval

    def __call__(self, results_dict: Dict) -> float:
        if self._call_count == 0:
            retval = self._mock_retval
        else:
            retval = results_dict["inverse_loss"]

        self._call_count += 1
        return retval


@pytest.mark.parametrize(
    ("max_accuracy_degradation", "final_compression_rate", "reference_final_metric"),
    (
        ({"maximal_relative_accuracy_degradation": 30.0}, -0.0923, 0.04942),
        ({"maximal_relative_accuracy_degradation": 1.0}, -0.0923, 0.04942),
        ({"maximal_absolute_accuracy_degradation": 0.0}, -0.0923, 0.04942),
        ({"maximal_absolute_accuracy_degradation": 0.10}, 0.2077, 0.09749),
    ),
)
def test_adaptive_compression_training_loop(
    max_accuracy_degradation,
    final_compression_rate,
    reference_final_metric,
    initial_training_phase_epochs=5,
    patience_epochs=3,
    steps_per_epoch=20,
    img_size=10,
):
    set_random_seed(42)
    model = get_simple_conv_regression_model(img_size)
    dataset = get_const_target_mock_regression_dataset(img_size=img_size, num_samples=steps_per_epoch)
    config = get_basic_magnitude_sparsity_config(input_sample_size=[1, img_size, img_size, 1])

    params = {
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
        "lr_reduction_factor": 1,
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "adaptive_compression_level", "params": params}}

    config.update(accuracy_aware_config)

    compress_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    compress_model.add_loss(compression_ctrl.loss)

    def inverse_loss(y_true, y_pred):
        return 1 / (1 + (y_true - y_pred) ** 2)

    compress_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=inverse_loss,
    )

    # The model is not actually fine-tuned at this stage. Will force the uncompressed
    # model accuracy to be reported higher than it is so that it fine-tunes at the same time
    # as it is being compressed.
    uncompressed_model_accuracy = 0.2
    mock_result_dict_to_val_metric_fn = MockResultFunctor(uncompressed_model_accuracy)

    statistics = compress_model.accuracy_aware_fit(
        dataset,
        compression_ctrl,
        uncompressed_model_accuracy=uncompressed_model_accuracy,
        nncf_config=config,
        callbacks=compression_callbacks,
        initial_epoch=0,
        steps_per_epoch=steps_per_epoch,
        result_dict_to_val_metric_fn=mock_result_dict_to_val_metric_fn,
    )
    assert statistics.compressed_accuracy == pytest.approx(reference_final_metric, 1e-4)
    assert statistics.compression_rate == pytest.approx(final_compression_rate, 1e-3)


@pytest.mark.parametrize(
    "max_accuracy_degradation",
    (
        ({"maximal_relative_accuracy_degradation": 30.0}),
        ({"maximal_relative_accuracy_degradation": 1.0}),
        ({"maximal_absolute_accuracy_degradation": 0.1}),
    ),
)
def test_early_exit_compression_training_loop(
    max_accuracy_degradation, maximal_total_epochs=100, steps_per_epoch=20, img_size=10
):
    set_random_seed(42)
    model = get_simple_conv_regression_model(img_size)
    dataset = get_const_target_mock_regression_dataset(img_size=img_size, num_samples=steps_per_epoch)

    config = get_basic_quantization_config(img_size)
    params = {
        "maximal_total_epochs": maximal_total_epochs,
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "early_exit", "params": params}}
    config.update(accuracy_aware_config)
    config = register_default_init_args(config, dataset, batch_size=1)
    compress_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    compress_model.add_loss(compression_ctrl.loss)

    def inverse_loss(y_true, y_pred):
        return 1 / (1 + (y_true - y_pred) ** 2)

    compress_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=inverse_loss,
    )

    # The model is not actually fine-tuned at this stage. Will force the uncompressed
    # model accuracy to be reported higher than it is so that it fine-tunes at the same time
    # as it is being compressed.
    uncompressed_model_accuracy = 0.2
    mock_result_dict_to_val_metric_fn = MockResultFunctor(uncompressed_model_accuracy)

    statistics = compress_model.accuracy_aware_fit(
        dataset,
        compression_ctrl,
        uncompressed_model_accuracy=uncompressed_model_accuracy,
        nncf_config=config,
        callbacks=compression_callbacks,
        initial_epoch=0,
        steps_per_epoch=steps_per_epoch,
        result_dict_to_val_metric_fn=mock_result_dict_to_val_metric_fn,
    )
    uncompressed_model_accuracy = statistics.uncompressed_accuracy
    compressed_model_accuracy = statistics.compressed_accuracy

    if "maximal_absolute_accuracy_degradation" in max_accuracy_degradation:
        assert (uncompressed_model_accuracy - compressed_model_accuracy) <= max_accuracy_degradation[
            "maximal_absolute_accuracy_degradation"
        ]
    else:
        assert (
            uncompressed_model_accuracy - compressed_model_accuracy
        ) / uncompressed_model_accuracy * 100 <= max_accuracy_degradation["maximal_relative_accuracy_degradation"]
