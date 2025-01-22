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

import numpy as np
import torch

from nncf.common.factory import TModel
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import TensorStatistic
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.transformations import leaf_module_insertion_transformation_builder
from nncf.tensor import Tensor
from nncf.torch.graph.transformations.commands import PTTargetPoint
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
    BACKEND: BackendType = BackendType.TORCH_FX
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

    @staticmethod
    def _get_statistic_collector_name(tp: PTTargetPoint, module_to_insert: torch.nn.Module) -> str:
        """
        Compouses unique statistic collector name according to given target point and module.

        :param tp: Given target point.
        :param module_to_insert: Given statistic collection module.
        :return: Unique statistic collector name according to given target point and module.
        """
        return "_".join(
            [
                tp.target_node_name,
                str(tp.input_port_id),
                str(tp.target_type.value),
                str(id(module_to_insert)),
            ]
        )

    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []

        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                for collectors in _statistic_point.algorithm_to_tensor_collectors.values():
                    for collector in collectors:
                        tp = _statistic_point.target_point
                        module_to_insert = TensorCollectorModule(collector)
                        target_module_name = self._get_statistic_collector_name(tp, module_to_insert)
                        transformation = leaf_module_insertion_transformation_builder(
                            module_to_insert, [tp], target_module_name
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

    def _get_statistics_key(self, statistics: TensorStatistic, target_point: PTTargetPoint) -> str:
        """
        Returns key of statistics.

        :param statistics: Statistics value.
        :param target_point: Statistics target point.
        :return: Statistics key.
        """
        target_point_id = f"{target_point.target_node_name}_{target_point.type}_{target_point.input_port_id}"
        return f"{statistics.__class__.__name__}_{target_point_id}"
