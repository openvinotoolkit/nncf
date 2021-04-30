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

from typing import Union

import tensorflow as tf

from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics, RBSparsityStatistics
from beta.nncf.tensorflow.callbacks.statistics_callback import StatisticsCallback


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
    Callback for logging sparsity compression statistics to tensorboard and stdout.
    """

    def _prepare_for_tensorboard(self, statistics: Union[MagnitudeSparsityStatistics, RBSparsityStatistics]):
        base_prefix = '2.compression/statistics'
        detailed_prefix = '3.compression_details/statistics'

        ms = statistics.model_statistics  # type: SparsifiedModelStatistics
        tensorboard_statistics = {
            f'{base_prefix}/sparsity_level': ms.sparsity_level,
            f'{base_prefix}/sparsity_level_for_layers': ms.sparsity_level_for_layers,
        }

        for ls in ms.sparsified_layers_summary:
            layer_name, sparsity_level = ls.name, ls.sparsity_level
            tensorboard_statistics[f'{detailed_prefix}/{layer_name}/sparsity_level'] = sparsity_level

        return tensorboard_statistics
