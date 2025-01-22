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

from typing import Dict

import torch

from nncf.common.factory import TModel
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.statistics import TensorStatistic
from nncf.tensor import Tensor
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.tensor_statistics.algo import create_register_input_hook


class PTStatisticsAggregator(StatisticsAggregator):
    BACKEND: BackendType = BackendType.TORCH
    HOOKS_GROUP_NAME = "statistics_hooks"

    def collect_statistics(self, model: NNCFNetwork, graph: NNCFGraph) -> None:
        with torch.no_grad():
            super().collect_statistics(model, graph)
        model.nncf.remove_hooks_group(self.HOOKS_GROUP_NAME)

    def _register_statistics(self, outputs: Dict[str, Tensor], statistic_points: StatisticPointsContainer) -> None:
        # PyTorch backend doesn't use outputs to register statistics
        return

    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []

        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                for collectors in _statistic_point.algorithm_to_tensor_collectors.values():
                    for collector in collectors:
                        transformation_commands.append(
                            PTInsertionCommand(
                                _statistic_point.target_point,
                                create_register_input_hook(collector=collector),
                                TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION,
                                hooks_group_name=self.HOOKS_GROUP_NAME,
                            )
                        )

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        # TODO: mirgate to experimental statistic collector and use common merging algorithm
        return statistic_points

    @staticmethod
    def _process_outputs(outputs: torch.Tensor) -> Dict[str, Tensor]:
        # PyTorch backend doesn't use outputs to register statistics
        return {}

    def _get_statistics_key(self, statistics: TensorStatistic, target_point: PTTargetPoint) -> str:
        """
        Returns key of statistics.

        :param statistics: Statistics value.
        :param target_point: Statistics target point.
        :return: Statistics key.
        """
        target_point_id = f"{target_point.target_node_name}_{target_point.type}_{target_point.input_port_id}"
        return f"{statistics.__class__.__name__}_{target_point_id}"
