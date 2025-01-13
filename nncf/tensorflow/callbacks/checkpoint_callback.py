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

from typing import Union

import tensorflow as tf


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """Callback to save the NNCF TF compression model with compression state."""

    def __init__(self, checkpoint: tf.train.Checkpoint, directory: str, save_freq: Union[str, int] = "epoch"):
        """
        :param checkpoint: The `tf.train.Checkpoint` instance to save and manage
            checkpoints for.
        :param directory: The path to a directory in which to write checkpoints.
        :param save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves.
            the model after each epoch and name number of the checkpoint correspond to epoch
            model was saved. When using integer, the callback saves the model at end of this many
            batches and name number of the checkpoint correspond to number of passed batches * `save_freq`.
        """
        super().__init__()
        self._last_batch_seen = 0
        self._batches_seen_since_last_saving = 0
        if save_freq != "epoch" and not isinstance(save_freq, int):
            raise ValueError("Unrecognized save_freq: {}".format(save_freq))

        self._checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory, None)
        self._save_freq = save_freq

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model()

    def on_epoch_end(self, epoch, logs=None):
        if self._save_freq == "epoch":
            self._save_model()

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self._save_freq == "epoch":
            return False

        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen

        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self._save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_model(self):
        self._checkpoint_manager.save()
