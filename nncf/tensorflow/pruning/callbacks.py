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

from nncf.common.statistics import NNCFStatistics
from nncf.tensorflow.callbacks.statistics_callback import StatisticsCallback


class PruningStatisticsCallback(StatisticsCallback):
    """
    Callback for logging pruning compression statistics to tensorboard and stdout.
    """

    def _prepare_for_tensorboard(self, nncf_stats: NNCFStatistics):
        base_prefix = '2.compression/statistics'
        detailed_prefix = '3.compression_details/statistics'

        ms = nncf_stats.filter_pruning.model_statistics
        tensorboard_stats = {
            f'{base_prefix}/pruning_level_for_model': ms.pruning_level,
            f'{base_prefix}/target_pruning_level': nncf_stats.filter_pruning.target_pruning_level,
        }

        for ls in ms.pruned_layers_summary:
            layer_name, pruning_level = ls.name, ls.filter_pruning_level
            tensorboard_stats[f'{detailed_prefix}/{layer_name}/pruning_level'] = pruning_level

        return tensorboard_stats
