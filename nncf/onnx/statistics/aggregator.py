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
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.tensor import ONNXNNCFTensor


class ONNXStatisticsAggregator(StatisticsAggregator):

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        self._nncf_graph = NNCFGraphFactory.create(model)
        self.nncf_input_node_next_onnx_nodes = {}
        for input_node in self._nncf_graph.get_input_nodes():
            next_nodes = self._nncf_graph.get_next_nodes(input_node)
            self.nncf_input_node_next_onnx_nodes[input_node.node_name] = [node.node_name for node in next_nodes]
        self._onnx_graph = ONNXGraph(model)
        self._registered_weights = set()
        super().collect_statistics(model)

    def _register_activation_statistic(self, statistic_point: StatisticPointsContainer,
                                       target_point: ONNXTargetPoint,
                                       node_name: str,
                                       outputs: Dict[str, np.ndarray]) -> None:
        port_id = target_point.port_id
        if NNCFGraphNodeType.INPUT_NODE in target_point.target_node_name:
            nncf_node_name = self._nncf_graph.get_node_by_name(target_point.target_node_name)
            onnx_nodes_after_input_node = [edge.to_node for edge in
                                           self._nncf_graph.get_output_edges(nncf_node_name)]
            for onnx_node_name in onnx_nodes_after_input_node:
                edge_name = self._onnx_graph.get_node_edge_names(onnx_node_name.node_name)['input'][port_id]
                statistic_point.register_tensor(outputs[edge_name])
        elif target_point.type == TargetType.POST_LAYER_OPERATION:
            edge_name = self._onnx_graph.get_node_edge_names(node_name)['output'][port_id]
            statistic_point.register_tensor(outputs[edge_name])
        elif target_point.type == TargetType.PRE_LAYER_OPERATION:
            edge_name = self._onnx_graph.get_node_edge_names(node_name)['input'][port_id]
            statistic_point.register_tensor(outputs[edge_name])

    def _register_weight_statistic(self, statistic_point: StatisticPointsContainer,
                                   target_point: ONNXTargetPoint) -> None:
        node = self._onnx_graph.get_node_by_name(target_point.target_node_name)
        weight_tensor = self._onnx_graph.get_weight_tensor(node)
        statistic_point.register_tensor(ONNXNNCFTensor(weight_tensor[1]))

    def _register_statistics(self,
                             outputs: Dict[str, ONNXNNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        for node_name, _statistic_points in statistic_points.items():
            for statistic_point in _statistic_points:
                target_point = statistic_point.target_point
                if target_point.type in [TargetType.PRE_LAYER_OPERATION,
                                         TargetType.POST_LAYER_OPERATION]:
                    self._register_activation_statistic(statistic_point,
                                                        target_point, node_name, outputs)
                elif target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
                    # Register constant only once because it does not change
                    # during inference
                    if target_point.target_node_name not in self._registered_weights:
                        self._register_weight_statistic(statistic_point, target_point)
                        self._registered_weights.add(target_point.target_node_name)
                else:
                    RuntimeError('The statistics should be collected only from the input of output edges of the node')

    def _get_transformation_layout_extra_outputs(self,
                                                 statistic_points: StatisticPointsContainer) -> TransformationLayout:
        def is_activation_point(statistic_point: StatisticPoint) -> bool:
            return not statistic_point.target_point.is_weight_target_point()

        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                if is_activation_point(_statistic_point):
                    transformation_commands.append(ONNXOutputInsertionCommand(_statistic_point.target_point,
                                                                              self.nncf_input_node_next_onnx_nodes))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, ONNXNNCFTensor]:
        return {n: ONNXNNCFTensor(v) for n, v in outputs.items()}
