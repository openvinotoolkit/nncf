# Copyright (c) 2023 Intel Corporation
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
import onnx

from nncf.common.factory import TModel
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.onnx.graph.node_utils import get_input_edge
from nncf.onnx.graph.node_utils import get_input_edges_mapping
from nncf.onnx.graph.onnx_helper import get_name_to_node_map
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.tensor import ONNXNNCFTensor


class ONNXStatisticsAggregator(StatisticsAggregator):
    def collect_statistics(self, model: onnx.ModelProto, graph: NNCFGraph) -> None:
        self.input_edges_mapping = get_input_edges_mapping(graph)
        self.node_mapping = get_name_to_node_map(model)
        self._registered_weights = set()
        super().collect_statistics(model, graph)

    def _register_statistics(
        self, outputs: Dict[str, ONNXNNCFTensor], statistic_points: StatisticPointsContainer
    ) -> None:
        for _statistic_points in statistic_points.values():
            for statistic_point in _statistic_points:
                target_point = statistic_point.target_point
                port_id = target_point.port_id
                if target_point.target_node_name in self.input_edges_mapping:  # Input case
                    edge_name = get_input_edge(
                        target_point.target_node_name,
                        self.input_edges_mapping,
                        self.node_mapping,
                    )
                elif target_point.type == TargetType.POST_LAYER_OPERATION:
                    node = self.node_mapping[target_point.target_node_name]
                    edge_name = node.output[port_id]
                elif target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
                    node = self.node_mapping[target_point.target_node_name]
                    edge_name = node.input[port_id]
                statistic_point.register_tensor(outputs[edge_name])

    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(
                    ONNXOutputInsertionCommand(_statistic_point.target_point, self.input_edges_mapping)
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
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, ONNXNNCFTensor]:
        return {n: ONNXNNCFTensor(v) for n, v in outputs.items()}
