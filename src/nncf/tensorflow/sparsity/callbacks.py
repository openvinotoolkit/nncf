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

import tensorflow as tf

from nncf.common.statistics import NNCFStatistics
from nncf.tensorflow.callbacks.statistics_callback import StatisticsCallback


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

    def _prepare_for_tensorboard(self, stats: NNCFStatistics):
        base_prefix = "2.compression/statistics"
        detailed_prefix = "3.compression_details/statistics"

        if stats.magnitude_sparsity:
            stats = stats.magnitude_sparsity
        else:
            stats = stats.rb_sparsity

        ms = stats.model_statistics
        tensorboard_stats = {
            f"{base_prefix}/sparsity_level_for_model": ms.sparsity_level,
            f"{base_prefix}/sparsity_level_for_sparsified_layers": ms.sparsity_level_for_layers,
            f"{base_prefix}/target_sparsity_level": stats.target_sparsity_level,
        }

        for ls in ms.sparsified_layers_summary:
            layer_name, sparsity_level = ls.name, ls.sparsity_level
            tensorboard_stats[f"{detailed_prefix}/{layer_name}/sparsity_level"] = sparsity_level

        return tensorboard_stats
