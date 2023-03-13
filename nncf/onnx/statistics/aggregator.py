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
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXInplaceOpInsertionCommand
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

    def _register_statistics(self, outputs: Dict[str, ONNXNNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            target_point = statistic_point.target_point
            port_id = target_point.port_id
            node_name = target_point.target_node_name
            target_edges_names = []
            if NNCFGraphNodeType.INPUT_NODE in target_point.target_node_name:
                nncf_node_name = self._nncf_graph.get_node_by_name(target_point.target_node_name)
                onnx_nodes_after_input_node = [edge.to_node for edge in
                                               self._nncf_graph.get_output_edges(nncf_node_name)]
                for onnx_node_name in onnx_nodes_after_input_node:
                    target_edges_names.append(
                        self._onnx_graph.get_node_edge_names(onnx_node_name.node_name)['input'][port_id])
            elif target_point.type == TargetType.POST_LAYER_OPERATION:
                target_edges_names.append(
                    self._onnx_graph.get_node_edge_names(node_name)['output'][port_id])
            elif target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
                target_edges_names.append(
                    self._onnx_graph.get_node_edge_names(node_name)['input'][port_id])

            for edge_name in target_edges_names:
                input_names = tensor_collector.get_output_names(edge_name, port_id)
                tensor_collector.register_inputs([outputs[name] for name in input_names])

    def _get_transformation_layout_extra_outputs(self,
                                                 statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            for op_fn in tensor_collector.get_inplace_fn():
                transformation_commands.append(
                    ONNXInplaceOpInsertionCommand(statistic_point.target_point,
                                                  self.nncf_input_node_next_onnx_nodes, op_fn))
            if tensor_collector.any_stat_out_of_place():
                transformation_commands.append(ONNXOutputInsertionCommand(statistic_point.target_point,
                                                                          self.nncf_input_node_next_onnx_nodes))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, ONNXNNCFTensor]:
        return {n: ONNXNNCFTensor(v) for n, v in outputs.items()}
