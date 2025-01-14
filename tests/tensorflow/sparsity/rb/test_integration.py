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

import os
from pathlib import Path

import pytest
import tensorflow as tf
from tensorflow.python.framework.config import disable_op_determinism
from tensorflow.python.framework.config import enable_op_determinism

from examples.tensorflow.common.callbacks import get_callbacks
from examples.tensorflow.common.callbacks import get_progress_bar
from nncf import NNCFConfig
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.tensorflow.helpers.callback_creation import create_compression_callbacks
from nncf.tensorflow.helpers.model_creation import create_compressed_model

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "mock_models" / "LeNet.h5"


@pytest.fixture(name="deterministic_mode", scope="module")
def deterministic_mode_fixture():
    tf.keras.utils.set_random_seed(1)
    enable_op_determinism()
    yield
    disable_op_determinism()


def get_basic_sparsity_config(
    model_size=4,
    input_sample_size=None,
    sparsity_init=0.02,
    sparsity_target=0.5,
    sparsity_target_epoch=2,
    sparsity_freeze_epoch=3,
    scheduler="polinomial",
):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]

    config = NNCFConfig()
    config.update(
        {
            "model": "basic_sparse_conv",
            "model_size": model_size,
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {
                "algorithm": "rb_sparsity",
                "sparsity_init": sparsity_init,
                "params": {
                    "schedule": scheduler,
                    "sparsity_target": sparsity_target,
                    "sparsity_target_epoch": sparsity_target_epoch,
                    "sparsity_freeze_epoch": sparsity_freeze_epoch,
                },
            },
        }
    )
    return config


def get_lenet_model():
    inp = tf.keras.Input((28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 5)(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(48, 5)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dense(84)(x)
    y = tf.keras.layers.Dense(10, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=y)


def train_lenet():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.transpose(tf.reshape(x_train, (-1, 1, 28, 28)), (0, 2, 3, 1))
    x_test = tf.transpose(tf.reshape(x_test, (-1, 1, 28, 28)), (0, 2, 3, 1))

    x_train = x_train / 255
    x_test = x_test / 255

    model = get_lenet_model()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(5e-4),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=16,
        validation_split=0.2,
        callbacks=tf.keras.callbacks.ReduceLROnPlateau(),
    )

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save(MODEL_PATH)


@pytest.mark.parametrize(
    "distributed", [False, pytest.param(True, marks=pytest.mark.nightly)], ids=["not_distributed", "distributed"]
)
def test_rb_sparse_target_lenet(distributed, deterministic_mode):
    if not os.path.exists(MODEL_PATH):
        train_lenet()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test, y_test = x_test[:2], y_test[:2]

    x_train = tf.transpose(tf.reshape(x_train, (-1, 1, 28, 28)), (0, 2, 3, 1))
    x_test = tf.transpose(tf.reshape(x_test, (-1, 1, 28, 28)), (0, 2, 3, 1))

    x_train = x_train / 255
    x_test = x_test / 255

    batch_size = 128
    if distributed:
        coeff = 3
        strategy = tf.distribute.MirroredStrategy()
    else:
        coeff = 1
        strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")

    tf.keras.backend.clear_session()
    with strategy.scope():
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset_train = dataset_train.with_options(options)
        dataset_test = dataset_test.with_options(options)

        model = get_lenet_model()
        model.load_weights(MODEL_PATH)

        freeze_epoch = 4
        config = get_basic_sparsity_config(
            sparsity_init=0.04,
            sparsity_target=0.3,
            sparsity_target_epoch=3,
            sparsity_freeze_epoch=freeze_epoch,
            scheduler="exponential",
        )

        compression_state_to_skip_init = {BaseCompressionAlgorithmController.BUILDER_STATE: {}}
        compress_algo, compress_model = create_compressed_model(model, config, compression_state_to_skip_init)
        compression_callbacks = create_compression_callbacks(compress_algo, log_tensorboard=True, log_dir="logdir/")

        sparse_algo = (
            compress_algo.child_ctrls[0]
            if isinstance(compress_algo, CompositeCompressionAlgorithmController)
            else compress_algo
        )

        class SparsityRateTestCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                target = sparse_algo.loss.target_sparsity_rate
                nncf_stats = sparse_algo.statistics()
                actual = nncf_stats.rb_sparsity.model_statistics.sparsity_level_for_layers
                print(f"target {target}, actual {actual}")
                if epoch + 1 <= freeze_epoch:
                    assert abs(actual - target) < 0.05
                else:
                    assert tf.cast(sparse_algo.loss.disabled, tf.bool)
                    assert tf.equal(sparse_algo.loss.calculate(), tf.constant(0.0))

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc@1"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="acc@5"),
            tf.keras.metrics.MeanMetricWrapper(loss_obj, name="ce_loss"),
            tf.keras.metrics.MeanMetricWrapper(compress_algo.loss, name="cr_loss"),
        ]

        compress_model.add_loss(compress_algo.loss)

        compress_model.compile(
            loss=loss_obj,
            optimizer=tf.keras.optimizers.Adam(5e-3 * coeff),
            metrics=metrics,
        )

    compress_model.fit(
        dataset_train,
        validation_data=dataset_test,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(),
            get_progress_bar(stateful_metrics=["loss"] + [metric.name for metric in metrics]),
            *get_callbacks(
                include_tensorboard=True,
                track_lr=False,
                profile_batch=0,
                initial_step=0,
                log_dir="logdir/",
                ckpt_dir="logdir/cpt/",
            ),
            compression_callbacks,
            SparsityRateTestCallback(),
        ],
    )
