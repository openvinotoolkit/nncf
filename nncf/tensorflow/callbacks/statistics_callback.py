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

from typing import Callable

import tensorflow as tf

from nncf.common.logging import nncf_logger
from nncf.common.statistics import NNCFStatistics


class StatisticsCallback(tf.keras.callbacks.Callback):
    """
    Callback for logging compression statistics to tensorboard and stdout.
    """

    def __init__(
        self,
        statistics_fn: Callable[[], NNCFStatistics],
        log_tensorboard: bool = True,
        log_text: bool = True,
        log_dir: str = None,
    ):
        """
        Initializes compression statistics callback.

        :param statistics_fn: A callable object that provides NNCF statistics.
        :param log_tensorboard: Whether to log statistics to tensorboard or not.
        :param log_text: Whether to log statistics to stdout.
        :param log_dir: The directory for tensorbard logging.
        """
        super().__init__()
        self._statistics_fn = statistics_fn
        self._log_tensorboard = log_tensorboard
        self._log_text = log_text
        self._file_writer = None
        if log_tensorboard:
            if log_dir is None:
                raise ValueError("log_dir must be specified if log_tensorboard is true.")

            self._file_writer = tf.summary.create_file_writer(log_dir + "/compression")

    def _dump_to_tensorboard(self, logs: dict, step: int):
        with self._file_writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=step)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        nncf_stats = self._statistics_fn()
        if self._log_tensorboard:
            self._dump_to_tensorboard(
                self._prepare_for_tensorboard(nncf_stats), self.model.optimizer.iterations.numpy()
            )
        if self._log_text:
            nncf_logger.info(nncf_stats.to_str())

    def on_train_begin(self, logs: dict = None):
        nncf_stats = self._statistics_fn()
        if self._log_tensorboard:
            self._dump_to_tensorboard(
                self._prepare_for_tensorboard(nncf_stats), self.model.optimizer.iterations.numpy()
            )
        if self._log_text:
            nncf_logger.info(nncf_stats.to_str())

    def on_train_end(self, logs: dict = None):
        if self._file_writer:
            self._file_writer.close()

    def _prepare_for_tensorboard(self, stats: NNCFStatistics):
        raise NotImplementedError(
            "StatisticsCallback class implementation must override the _prepare_for_tensorboard method."
        )
