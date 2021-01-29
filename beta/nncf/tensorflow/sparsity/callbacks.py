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

import tensorflow as tf

from beta.nncf.helpers.utils import print_statistics
from beta.nncf.tensorflow.sparsity.utils import convert_raw_to_printable
from beta.nncf.tensorflow.sparsity.utils import prepare_for_tensorboard


class UpdateMask(tf.keras.callbacks.Callback):
    def __init__(self, scheduler):
        super().__init__()
        self._scheduler = scheduler

    def on_train_batch_begin(self, batch, logs=None):
        self._scheduler.step()

    def on_epoch_begin(self, epoch, logs=None):
        self._scheduler.epoch_step(epoch)


class SparsityStatistics(tf.keras.callbacks.Callback):
    """
    Callback for logging sparsity compression statistics to tensorboard and stdout
    """
    def __init__(self, raw_statistics_fn, log_tensorboard=True, log_text=True, log_dir=None):
        """
        Arguments:
            `raw_statistics_fn` - callable to evaluate raw sparsity compression statistics,
            `log_tensorboard` - whether to log statistics to tensorboard or not,
            `log_text` - whether to log statistics to stdout,
            `log_dir` -  the directory for tensorbard logging.
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

    def _dump_to_tensorboard(self, logs, step):
        with self._file_writer.as_default(): # pylint: disable=E1129
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=step)

    def on_epoch_end(self, epoch, logs=None):
        raw_statistics = self._raw_statistics_fn()
        if self._log_tensorboard:
            self._dump_to_tensorboard(prepare_for_tensorboard(raw_statistics),
                                      self.model.optimizer.iterations.numpy())
        if self._log_text:
            print_statistics(convert_raw_to_printable(raw_statistics))

    def on_train_end(self, logs=None):
        if self._file_writer:
            self._file_writer.close()
