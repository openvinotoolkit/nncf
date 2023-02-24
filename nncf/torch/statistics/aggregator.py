"""
 Copyright (c) 2023 Intel Corporation
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

import torch
import numpy as np

from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.graph.transformations.commands import PTInsertionCommand


class PTStatisticsAggregator(StatisticsAggregator):
    def collect_statistics(self, model: NNCFNetwork) -> None:
        with torch.no_grad():
            with model.temporary_clean_view() as intermediate_model:
                super().collect_statistics(intermediate_model)

    def _register_statistics(self,
                             outputs: Dict[str, PTNNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        return

    def _get_transformation_layout_extra_outputs(self,
                                                 statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                for collectors in _statistic_point.algorithm_to_tensor_collectors.values():
                    for collector in collectors:
                        transformation_commands.append(PTInsertionCommand(
                            _statistic_point.target_point,
                            collector.register_input,
                            TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, PTNNCFTensor]:
        return outputs
