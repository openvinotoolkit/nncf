# Copyright (c) 2024 Intel Corporation
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

import numpy as np
import torch

from nncf.common.factory import TModel
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.transformations import leaf_module_insertion_transformation_builder
from nncf.tensor import Tensor
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.return_types import maybe_get_values_from_torch_return_type


class TensorCollectorModule(torch.nn.Module):
    """
    torch.nn.Module which calls given collector in forward
    """

    def __init__(self, collector: TensorCollector):
        super().__init__()
        self._collector = collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Register inputs hook function.

        :parameter x: tensor to register in hook.
        :return: tensor to register in hook.
        """
        x_unwrapped = maybe_get_values_from_torch_return_type(x)
        self._collector.register_input_for_all_reducers(Tensor(x_unwrapped))
        return x


class FXStatisticsAggregator(StatisticsAggregator):
    HOOKS_GROUP_NAME = "statistics_hooks"

    def collect_statistics(self, model: NNCFNetwork, graph: NNCFGraph) -> None:
        with torch.no_grad():
            super().collect_statistics(model, graph)
        # All statistics are collected as a dead code,
        # so eliminate dead core removed statistcs collector
        # from the target model. No additional code required
        # for that, horay!
        model.graph.eliminate_dead_code()
        model.recompile()

    def _register_statistics(self, outputs: Dict[str, Tensor], statistic_points: StatisticPointsContainer) -> None:
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
                        transformation = leaf_module_insertion_transformation_builder(
                            TensorCollectorModule(collector), [_statistic_point.target_point]
                        )
                        transformation_commands.append(
                            FXApplyTransformationCommand(
                                transformation, TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION
                            )
                        )

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        # TODO(dlyakhov): mirgate to experimental statistic collector and use common merging algorithm
        return statistic_points

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        return outputs
