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

from typing import Dict
from functools import singledispatch

from nncf.common.statistics import NNCFStatistics
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.sparsity.statistics import ConstSparsityStatistics


def prepare_for_tensorboard(nncf_stats: NNCFStatistics) -> Dict[str, float]:
    """
    Extracts scalar values from NNCF statistics for its reporting to the TensorBoard.

    :param nncf_stats: NNCF Statistics.
    :return: A dict storing name and value of the scalar.
    """
    tensorboard_stats = {}
    for algorithm_name, stats in nncf_stats:
        tensorboard_stats.update(convert_to_dict(stats, algorithm_name))

    return tensorboard_stats


@singledispatch
def convert_to_dict(stats, algorithm_name: str):
    return {}


@convert_to_dict.register(FilterPruningStatistics)
def _(stats, algorithm_name):
    tensorboard_stats = {
        f'{algorithm_name}/pruning_level_for_model': stats.model_statistics.pruning_level,
        f'{algorithm_name}/flops_pruning_level': stats.flops_pruning_level,
        f'{algorithm_name}/target_pruning_level': stats.target_pruning_level,
    }
    return tensorboard_stats


@convert_to_dict.register(MagnitudeSparsityStatistics)
@convert_to_dict.register(RBSparsityStatistics)
@convert_to_dict.register(ConstSparsityStatistics)
def _(stats, algorithm_name):
    tensorboard_stats = {
        f'{algorithm_name}/sparsity_level_for_model': stats.model_statistics.sparsity_level,
        f'{algorithm_name}/sparsity_level_for_sparsified_layers': stats.model_statistics.sparsity_level_for_layers,
    }

    target_sparsity_level = getattr(stats, 'target_sparsity_level', None)
    if target_sparsity_level is not None:
        tensorboard_stats[f'{algorithm_name}/target_sparsity_level'] = target_sparsity_level

    return tensorboard_stats
