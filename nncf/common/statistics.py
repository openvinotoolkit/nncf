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

from dataclasses import dataclass
from dataclasses import fields
from typing import Iterator, Optional, Tuple

from nncf.api.statistics import Statistics
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.quantization.statistics import QuantizationStatistics
from nncf.common.sparsity.statistics import ConstSparsityStatistics
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.utils.api_marker import api


@api()
@dataclass
class NNCFStatistics:
    """
    Groups statistics for all available NNCF compression algorithms.
    Statistics are present only if the algorithm has been started.
    """

    const_sparsity: Optional[ConstSparsityStatistics] = None
    filter_pruning: Optional[FilterPruningStatistics] = None
    magnitude_sparsity: Optional[MagnitudeSparsityStatistics] = None
    movement_sparsity: Optional[MovementSparsityStatistics] = None
    quantization: Optional[QuantizationStatistics] = None
    rb_sparsity: Optional[RBSparsityStatistics] = None

    def register(self, algorithm_name: str, stats: Statistics) -> None:
        """
        Registers statistics for the algorithm.

        :param algorithm_name: Name of the algorithm. Should be one of the following
            * const_sparsity
            * filter_pruning
            * magnitude_sparsity
            * movement_sparsity
            * quantization
            * rb_sparsity

        :param stats: Statistics of the algorithm.
        """
        available_algorithms = [f.name for f in fields(self)]
        if algorithm_name not in available_algorithms:
            raise ValueError(
                f"Can not register statistics for the algorithm. Unknown name of the algorithm: {algorithm_name}."
            )

        setattr(self, algorithm_name, stats)

    def to_str(self) -> str:
        pretty_string = "\n\n".join([x[1].to_str() for x in self])
        return pretty_string

    def __iter__(self) -> Iterator[Tuple[str, Statistics]]:
        return iter([(f.name, getattr(self, f.name)) for f in fields(self) if getattr(self, f.name) is not None])
