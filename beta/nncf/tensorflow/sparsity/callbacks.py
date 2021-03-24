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

from beta.nncf.tensorflow.callbacks.statistics_callback import StatisticsCallback
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


class SparsityStatisticsCallback(StatisticsCallback):
    """
    Callback for logging sparsity compression statistics to tensorboard and stdout
    """

    def _prepare_for_tensorboard(self, raw_statistics: dict):
        prefix = 'sparsity'
        rate_abbreviation = 'SR'
        return prepare_for_tensorboard(raw_statistics, prefix, rate_abbreviation)

    def _convert_raw_to_printable(self, raw_statistics: dict):
        prefix = 'sparsity'
        header = ['Name', 'Weight\'s Shape', 'SR', '% weights']
        return convert_raw_to_printable(raw_statistics, prefix, header)
