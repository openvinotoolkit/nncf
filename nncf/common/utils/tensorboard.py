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

from nncf.api.compression import Statistics
from nncf.api.composite_compression import CompositeStatistics
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.sparsity.statistics import ConstSparsityStatistics


def prepare_for_tensorboard(statistics: Statistics) -> Dict[str, float]:
    tensorboard_stats = {}

    if isinstance(statistics, CompositeStatistics):
        items = statistics.child_statistics
    else:
        items = [statistics]

    for item in items:
        tensorboard_stats.update(convert_to_dict(item))

    return tensorboard_stats


@singledispatch
def convert_to_dict(statistics):
    return {}


@convert_to_dict.register(FilterPruningStatistics)
def _(statistics):
    tensorboard_stats = {
        'pruning_level_for_model': statistics.model_statistics.pruning_level,
        'flops_pruning_level': statistics.flops_pruning_level,
    }
    return tensorboard_stats


@convert_to_dict.register(MagnitudeSparsityStatistics)
@convert_to_dict.register(RBSparsityStatistics)
@convert_to_dict.register(ConstSparsityStatistics)
def _(statistics):
    tensorboard_stats = {
        'sparsity_level_for_model': statistics.model_statistics.sparsity_level,
        'sparsity_level_for_sparsified_layers': statistics.model_statistics.sparsity_level_for_layers,
    }
    return tensorboard_stats
