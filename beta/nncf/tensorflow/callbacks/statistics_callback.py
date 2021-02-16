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
from typing import Callable

import tensorflow as tf

from beta.nncf.helpers.utils import print_statistics


class StatisticsCallback(tf.keras.callbacks.Callback):
    """
    Callback for logging compression statistics to tensorboard and stdout
    """

    def __init__(self,
                 raw_statistics_fn: Callable[[], dict],
                 log_tensorboard: bool = True,
                 log_text: bool = True,
                 log_dir: str = None):
        """
        :param raw_statistics_fn: callable to evaluate raw sparsity compression statistics
        :param log_tensorboard: whether to log statistics to tensorboard or not
        :param log_text: whether to log statistics to stdout
        :param log_dir: the directory for tensorbard logging
        """
        super().__init__()
        self._raw_statistics_fn = raw_statistics_fn
        self._log_tensorboard = log_tensorboard
        self._log_text = log_text
        self._file_writer = None
        if log_tensorboard:
            if log_dir is None:
                raise RuntimeError('log_dir must be specified if log_tensorboard is true.')
            # pylint: disable=no-member
            self._file_writer = tf.summary.create_file_writer(log_dir + '/compression')

    def _dump_to_tensorboard(self, logs: dict, step: int):
        with self._file_writer.as_default(): # pylint: disable=E1129
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=step)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        raw_statistics = self._raw_statistics_fn()
        if self._log_tensorboard:
            self._dump_to_tensorboard(self._prepare_for_tensorboard(raw_statistics),
                                      self.model.optimizer.iterations.numpy())
        if self._log_text:
            print_statistics(self._convert_raw_to_printable(raw_statistics))

    def on_train_end(self, logs: dict = None):
        if self._file_writer:
            self._file_writer.close()

    def _prepare_for_tensorboard(self, raw_statistics: dict):
        raise NotImplementedError

    def _convert_raw_to_printable(self, raw_statistics: dict):
        raise NotImplementedError
