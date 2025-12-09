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
from typing import Any

from nncf.api.statistics import Statistics
from nncf.common.statistics import NNCFStatistics


def prepare_for_tensorboard(nncf_stats: NNCFStatistics) -> dict[str, float]:
    """
    Extracts scalar values from NNCF statistics for its reporting to the TensorBoard.

    :param nncf_stats: NNCF Statistics.
    :return: A dict storing name and value of the scalar.
    """
    tensorboard_stats: dict[str, float] = {}
    for algorithm_name, stats in nncf_stats:
        tensorboard_stats.update(convert_to_dict(stats, algorithm_name))

    return tensorboard_stats


@singledispatch
def convert_to_dict(
    stats: Statistics,
    algorithm_name: str,
) -> dict[Any, Any]:
    return {}
