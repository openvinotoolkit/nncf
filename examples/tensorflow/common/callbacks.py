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

from typing import Any, List, MutableMapping

import tensorflow as tf
from absl import logging

from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback


def get_callbacks(include_tensorboard: bool = True,
                  track_lr: bool = True,
                  write_model_weights: bool = True,
                  initial_step: int = 0,
                  model_dir: str = None,
                  ckpt_dir: str = None,
                  checkpoint: tf.train.Checkpoint = None) -> List[tf.keras.callbacks.Callback]:
    """Get all callbacks."""
    model_dir = model_dir or ''
    ckpt_dir = ckpt_dir or model_dir
    callbacks = []
    if checkpoint:
        callbacks.append(CheckpointManagerCallback(checkpoint, ckpt_dir))
    if include_tensorboard:
        callbacks.append(
            CustomTensorBoard(
                log_dir=model_dir,
                track_lr=track_lr,
                initial_step=initial_step,
                write_images=write_model_weights))
    return callbacks


def get_progress_bar(stateful_metrics: list = None):
    # TODO: This is a workaround for tf2.4.0. Remove when fixed
    stateful_metrics = stateful_metrics or []
    stateful_metrics.extend(['val_' + metric_name for metric_name in stateful_metrics])
    return tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                            stateful_metrics=stateful_metrics)


def get_scalar_from_tensor(t: tf.Tensor) -> int:
    """Utility function to convert a Tensor to a scalar."""
    t = tf.keras.backend.get_value(t)
    if callable(t):
        return t()
    return t


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A customized TensorBoard callback that tracks additional datapoints.

    Metrics tracked:
    - Global learning rate

    Attributes:
      log_dir: The path of the directory where to save the log files to be parsed
        by TensorBoard.
      track_lr: `bool`, Whether or not to track the global learning rate.
      initial_step: The initial step, used for preemption recovery.
      **kwargs: Additional arguments for backwards compatibility. Possible key is
        `period`.
    """

    def __init__(self,
                 log_dir: str,
                 track_lr: bool = False,
                 initial_step: int = 0,
                 **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.step = initial_step
        self._track_lr = track_lr

    def on_train_batch_begin(self,
                             epoch: int, # pylint: disable=W0613
                             logs: MutableMapping[str, Any] = None) -> None:
        self.step += 1
        logs = logs or {}
        logs.update(self._calculate_metrics())
        super().on_train_batch_begin(self.step, logs)

    def on_epoch_begin(self,
                       epoch: int,
                       logs: MutableMapping[str, Any] = None) -> None:
        if logs is None:
            logs = {}
        metrics = self._calculate_metrics()
        logs.update(metrics)
        for k, v in metrics.items():
            logging.info('Current %s: %f', k, v)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self,
                     epoch: int,
                     logs: MutableMapping[str, Any] = None) -> None:
        logs = logs or {}
        metrics = self._calculate_metrics()
        logs_ = {}
        logs_.update(logs)
        logs_.update(metrics)
        with tf.name_scope('1.train_validation'):
            self._log_epoch_metrics(self.step, logs_)
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(self.step)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(self.step)

    def _calculate_metrics(self) -> MutableMapping[str, Any]:
        logs = {}
        if self._track_lr:
            logs['learning_rate'] = self._calculate_lr()
        return logs

    def _calculate_lr(self) -> int:
        """Calculates the learning rate given the current step."""
        return get_scalar_from_tensor(self._get_base_optimizer().lr(self.step))

    def _get_base_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Get the base optimizer used by the current model."""

        optimizer = self.model.optimizer

        # The optimizer might be wrapped by another class, so unwrap it
        while hasattr(optimizer, '_optimizer'):
            optimizer = optimizer._optimizer  # pylint:disable=protected-access

        return optimizer
