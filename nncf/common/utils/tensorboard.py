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

from functools import singledispatch
from typing import Any, Dict, Union

from nncf.api.statistics import Statistics
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.sparsity.statistics import ConstSparsityStatistics
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.statistics import NNCFStatistics


def prepare_for_tensorboard(nncf_stats: NNCFStatistics) -> Dict[str, float]:
    """
    Extracts scalar values from NNCF statistics for its reporting to the TensorBoard.

    :param nncf_stats: NNCF Statistics.
    :return: A dict storing name and value of the scalar.
    """
    tensorboard_stats: Dict[str, float] = {}
    for algorithm_name, stats in nncf_stats:
        tensorboard_stats.update(convert_to_dict(stats, algorithm_name))

    return tensorboard_stats


@singledispatch
def convert_to_dict(
    stats: Statistics,
    algorithm_name: str,
) -> Dict[Any, Any]:
    return {}


@convert_to_dict.register(FilterPruningStatistics)
def _(stats: FilterPruningStatistics, algorithm_name: str) -> Dict[str, float]:
    tensorboard_stats = {
        f"{algorithm_name}/algo_current_pruning_level": stats.current_pruning_level,
        f"{algorithm_name}/model_FLOPS_pruning_level": stats.model_statistics.flops_pruning_level,
        f"{algorithm_name}/model_params_pruning_level": stats.model_statistics.params_pruning_level,
        f"{algorithm_name}/model_filters_pruning_level": stats.model_statistics.filter_pruning_level,
    }
    return tensorboard_stats


@convert_to_dict.register(MagnitudeSparsityStatistics)
@convert_to_dict.register(RBSparsityStatistics)
@convert_to_dict.register(ConstSparsityStatistics)
def _(
    stats: Union[MagnitudeSparsityStatistics, RBSparsityStatistics, ConstSparsityStatistics], algorithm_name: str
) -> Dict[str, float]:
    tensorboard_stats = {
        f"{algorithm_name}/sparsity_level_for_model": stats.model_statistics.sparsity_level,
        f"{algorithm_name}/sparsity_level_for_sparsified_layers": stats.model_statistics.sparsity_level_for_layers,
    }

    target_sparsity_level = getattr(stats, "target_sparsity_level", None)
    if target_sparsity_level is not None:
        tensorboard_stats[f"{algorithm_name}/target_sparsity_level"] = target_sparsity_level

    return tensorboard_stats


@convert_to_dict.register(MovementSparsityStatistics)
def _(stats: MovementSparsityStatistics, algorithm_name: str) -> Dict[str, float]:
    tensorboard_stats = {
        f"{algorithm_name}/model_sparsity": stats.model_statistics.sparsity_level,
        f"{algorithm_name}/linear_layer_sparsity": stats.model_statistics.sparsity_level_for_layers,
        f"{algorithm_name}/importance_threshold": stats.importance_threshold,
        f"{algorithm_name}/importance_regularization_factor": stats.importance_regularization_factor,
    }
    return tensorboard_stats
