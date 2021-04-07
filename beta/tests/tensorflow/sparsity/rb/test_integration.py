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

import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path
import os

from beta.nncf import NNCFConfig
from beta.nncf.helpers.callback_creation import create_compression_callbacks
from beta.tests.tensorflow.helpers import create_compressed_model_and_algo_for_test


MODEL_PATH = Path(__file__).parent.parent.parent / 'data' / 'mock_models' / 'LeNet.h5'


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
                "sparsity_init": sparsity_init,
                "params":
                    {
                        "schedule": "polynomial",
                        "sparsity_target": sparsity_target,
                        "sparsity_target_epoch": sparsity_target_epoch,
                        "sparsity_freeze_epoch": sparsity_freeze_epoch
                    },
            }
    })
    return config


def train_lenet():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.transpose(tf.reshape(x_train, (-1, 1, 28, 28)), (0, 2, 3, 1))
    x_test = tf.transpose(tf.reshape(x_test, (-1, 1, 28, 28)), (0, 2, 3, 1))

    x_train = x_train / 255
    x_test = x_test / 255

    inp = tf.keras.Input((28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 5)(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(48, 5)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dense(84)(x)
    y = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=y)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(5e-4),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=64, epochs=16, validation_split=0.2,
              callbacks=tf.keras.callbacks.ReduceLROnPlateau())

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save(MODEL_PATH)


def test_rb_sparse_target_lenet():
    if not os.path.exists(MODEL_PATH):
        train_lenet()
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    x_train = tf.transpose(tf.reshape(x_train, (-1, 1, 28, 28)), (0, 2, 3, 1))
    x_train = x_train / 255

    model = tf.keras.models.load_model(MODEL_PATH)

    freeze_epoch = 4
    config = get_basic_sparsity_config(sparsity_init=0.05, sparsity_target=0.3,
                                       sparsity_target_epoch=3, sparsity_freeze_epoch=freeze_epoch)
    compress_model, compress_algo = create_compressed_model_and_algo_for_test(model, config)
    compression_callbacks = create_compression_callbacks(compress_algo, log_tensorboard=True, log_dir='logdir/')

    class SparsityRateTestCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            target = compress_algo.loss.target_sparsity_rate
            actual = compress_algo.raw_statistics()['sparsity_rate_for_sparsified_modules']
            print(f'target {target}, actual {actual}')
            if epoch <= freeze_epoch:
                assert abs(actual - target) < 0.05
            else:
                assert target == 0.

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    metrics = [loss_obj,
               tfa.metrics.MeanMetricWrapper(compress_algo.loss,
                                             name='rb_loss')]

    compress_model.add_loss(compress_algo.loss)

    compress_model.compile(
        loss=loss_obj,
        optimizer=tf.keras.optimizers.Adam(5e-3),
        metrics=metrics,
    )

    compress_model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2,
                       callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),
                                  compression_callbacks,
                                  SparsityRateTestCallback()])
