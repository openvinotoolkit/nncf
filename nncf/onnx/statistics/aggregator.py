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

import numpy as np
import onnx

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.tensor import ONNXNNCFTensor


class ONNXStatisticsAggregator(StatisticsAggregator):

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        self._nncf_graph = NNCFGraphFactory.create(model)
        self._onnx_graph = ONNXGraph(model)
        super().collect_statistics(model)

    def _register_statistics(self,
                             outputs: Dict[str, ONNXNNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        for node_name, _statistic_points in statistic_points.items():
            for statistic_point in _statistic_points:
                port_id = statistic_point.target_point.port_id
                if NNCFGraphNodeType.INPUT_NODE in statistic_point.target_point.target_node_name:
                    nncf_node_name = self._nncf_graph.get_node_by_name(statistic_point.target_point.target_node_name)
                    onnx_nodes_after_input_node = [edge.to_node for edge in
                                                   self._nncf_graph.get_output_edges(nncf_node_name)]
                    for onnx_node_name in onnx_nodes_after_input_node:
                        edge_name = self._onnx_graph.get_node_edge_names(onnx_node_name.node_name)['input'][port_id]
                        statistic_point.register_tensor(outputs[edge_name])
                elif statistic_point.target_point.type == TargetType.POST_LAYER_OPERATION:
                    edge_name = self._onnx_graph.get_node_edge_names(node_name)['output'][port_id]
                    statistic_point.register_tensor(outputs[edge_name])
                elif statistic_point.target_point.type == TargetType.PRE_LAYER_OPERATION:
                    edge_name = self._onnx_graph.get_node_edge_names(node_name)['input'][port_id]
                    statistic_point.register_tensor(outputs[edge_name])
                else:
                    RuntimeError('The statistics should be collected only from the input of output edges of the node')

    @staticmethod
    def _get_transformation_layout_extra_outputs(statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(ONNXOutputInsertionCommand(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, ONNXNNCFTensor]:
        return {n: ONNXNNCFTensor(v) for n, v in outputs.items()}
