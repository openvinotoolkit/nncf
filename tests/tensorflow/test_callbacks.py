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
import shutil
import tempfile

import pytest
import tensorflow as tf

from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback
from nncf.tensorflow.helpers.callback_creation import create_compression_callbacks
from nncf.tensorflow.helpers.model_creation import create_compressed_model
from nncf.tensorflow.utils.state import TFCompressionState
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.helpers import get_empty_config

REF_CKPT_DIR = {
    "epoch": [
        "checkpoint",
        "ckpt-1.data-00000-of-00001",
        "ckpt-1.index",
        "ckpt-2.data-00000-of-00001",
        "ckpt-2.index",
        "ckpt-3.data-00000-of-00001",
        "ckpt-3.index",
        "ckpt-4.data-00000-of-00001",
        "ckpt-4.index",
        "ckpt-5.data-00000-of-00001",
        "ckpt-5.index",
    ],
    2: [
        "checkpoint",
        "ckpt-1.data-00000-of-00001",
        "ckpt-1.index",
        "ckpt-10.data-00000-of-00001",
        "ckpt-10.index",
        "ckpt-2.data-00000-of-00001",
        "ckpt-2.index",
        "ckpt-3.data-00000-of-00001",
        "ckpt-3.index",
        "ckpt-4.data-00000-of-00001",
        "ckpt-4.index",
        "ckpt-5.data-00000-of-00001",
        "ckpt-5.index",
        "ckpt-6.data-00000-of-00001",
        "ckpt-6.index",
        "ckpt-7.data-00000-of-00001",
        "ckpt-7.index",
        "ckpt-8.data-00000-of-00001",
        "ckpt-8.index",
        "ckpt-9.data-00000-of-00001",
        "ckpt-9.index",
    ],
}


def get_simple_compressed_model(compression_state=None):
    model = get_basic_conv_test_model()
    config = get_empty_config()
    config.update({"compression": {"algorithm": "magnitude_sparsity"}})
    compression_ctrl, model = create_compressed_model(model, config, compression_state=compression_state)
    return compression_ctrl, model


@pytest.mark.parametrize("save_freq", ["epoch", 2], ids=["per_epoch", "per_n_steps"])
def test_checkpoint_callback_make_checkpoints(save_freq):
    compression_ctrl, model = get_simple_compressed_model()
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    dataset_len = 8

    dummy_x = tf.random.normal((dataset_len,) + model.input_shape[1:])
    dummy_y = tf.random.normal((dataset_len,) + model.output_shape[1:])

    model.compile(loss=tf.losses.CategoricalCrossentropy())

    ckpt_path = tempfile.mkdtemp()
    ckpt = tf.train.Checkpoint(model=model, compression_state=TFCompressionState(compression_ctrl))
    model.fit(
        dummy_x,
        dummy_y,
        epochs=5,
        batch_size=2,
        callbacks=[CheckpointManagerCallback(ckpt, ckpt_path, save_freq), *compression_callbacks],
    )

    assert sorted(os.listdir(ckpt_path)) == REF_CKPT_DIR[save_freq]

    new_compression_ctrl, new_model = get_simple_compressed_model()
    new_ckpt = tf.train.Checkpoint(model=new_model, compression_state=TFCompressionState(new_compression_ctrl))
    new_ckpt.restore(tf.train.latest_checkpoint(ckpt_path))
    assert new_compression_ctrl.get_state() == compression_ctrl.get_state()
    assert tf.reduce_all([tf.reduce_all(w_new == w) for w_new, w in zip(new_model.weights, model.weights)])

    shutil.rmtree(ckpt_path)
