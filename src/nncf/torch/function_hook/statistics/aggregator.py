# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
from torch import nn

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.collectors import TensorCollector
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.data.dataset import Dataset
from nncf.tensor import Tensor
from nncf.torch.function_hook.commands import PT2InsertionCommand
from nncf.torch.function_hook.hook_storage import RemovableHookHandle
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.graph.transformations.commands import PTTargetPoint


class StatisticCollectorModule(nn.Module):
    def __init__(self, collector: TensorCollector):
        super().__init__()
        self.collector = collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.collector.register_input_for_all_reducers(Tensor(x))
        return x


class PT2StatisticsAggregator(StatisticsAggregator):
    BACKEND: BackendType = BackendType.TORCH
    HOOKS_GROUP_NAME = "statistics_hooks"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.hook_handles: list[RemovableHookHandle] = []

    def collect_statistics(self, model: GraphModelWrapper, graph: NNCFGraph) -> None:  # type: ignore[override]
        with torch.no_grad():
            super().collect_statistics(model, graph)

        for hook_handle in self.hook_handles:
            hook_handle.remove()

    def _register_statistics(self, outputs: Any, statistic_points: StatisticPointsContainer) -> None:
        # PyTorch backend doesn't use outputs to register statistics
        return

    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        transformation_layout = TransformationLayout()

        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                target_point = _statistic_point.target_point
                for collectors in _statistic_point.algorithm_to_tensor_collectors.values():
                    for collector in collectors:
                        command = PT2InsertionCommand(
                            target_points=[target_point],
                            hook_module=StatisticCollectorModule(collector),
                            handle_storage=self.hook_handles,
                        )
                        transformation_layout.register(command)

        return transformation_layout

    @staticmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer,
        model: GraphModelWrapper,  # type: ignore[override]
        graph: NNCFGraph,
    ) -> StatisticPointsContainer:
        # TODO: migrate to experimental statistic collector and use common merging algorithm
        return statistic_points

    @staticmethod
    def _process_outputs(outputs: torch.Tensor) -> dict[str, Tensor]:
        # PyTorch backend doesn't use outputs to register statistics
        return {}

    def _get_statistics_key(self, statistics: TensorStatistic, target_point: TargetPoint) -> str:
        """
        Returns key of statistics.

        :param statistics: Statistics value.
        :param target_point: Statistics target point.
        :return: Statistics key.
        """
        if not isinstance(target_point, PTTargetPoint):
            msg = f"Unexpected target point type: {type(target_point)}"
            raise nncf.InternalError(msg)
        target_point_id = f"{target_point.target_node_name}_{target_point.type}_{target_point.input_port_id}"
        return f"{statistics.__class__.__name__}_{target_point_id}"
