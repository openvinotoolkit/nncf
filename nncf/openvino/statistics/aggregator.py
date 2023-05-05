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

from collections import defaultdict
from typing import Dict

import numpy as np
import openvino.runtime as ov

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.common.tensor_statistics.collectors import MergedTensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.transformations.commands import OVInplaceFnInsertionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.tensor import OVNNCFTensor


class OVStatisticsAggregator(StatisticsAggregator):
    def collect_statistics(self, model: ov.Model) -> None:
        self._name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        super().collect_statistics(model)

    def _register_statistics(
        self, outputs: Dict[str, OVNNCFTensor], statistic_points: StatisticPointsContainer
    ) -> None:
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            target_point = statistic_point.target_point
            node_name = target_point.target_node_name
            port_id = target_point.port_id
            if target_point.type == TargetType.POST_LAYER_OPERATION:
                stat_node_name = node_name
            elif target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
                node = self._name_to_node_mapping[node_name]
                output = node.input_value(port_id)
                stat_node_name = output.get_node().get_friendly_name()
                port_id = output.get_index()
            else:
                RuntimeError(f"Unsupported target point type for statistic aggregator:" f" {target_point.type}")

            input_info = tensor_collector.get_output_info(stat_node_name, port_id)
            target_inputs = TensorCollector.get_tensor_collector_inputs(outputs, input_info)
            tensor_collector.register_inputs(target_inputs)

    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            for op_fn, fn_out_port_id in tensor_collector.get_inplace_fn_info():
                transformation_commands.append(
                    OVInplaceFnInsertionCommand(statistic_point.target_point, op_fn, fn_out_port_id)
                )
            if tensor_collector.any_stat_out_of_place():
                transformation_commands.append(OVOutputInsertionCommand(statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    # TODO(dlyakhov) Move this to common part
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: ov.Model
    ) -> StatisticPointsContainer:
        nncf_graph = GraphConverter.create_nncf_graph(model)
        merged_statistic_points = StatisticPointsContainer()
        target_type_to_tensor_collector_map = defaultdict(lambda: defaultdict(list))
        for target_node_name, _statistic_points in statistic_points.data.items():
            for statistic_point in _statistic_points:
                target_point = statistic_point.target_point
                if target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
                    node = nncf_graph.get_node_by_name(target_node_name)
                    target_input_edge = nncf_graph.get_input_edges(node)[target_point.port_id]

                    target_type = TargetType.POST_LAYER_OPERATION
                    _target_node_name = target_input_edge.from_node.node_name
                    port_id = target_input_edge.output_port_id
                else:
                    target_type = statistic_point.target_point.type
                    _target_node_name = target_point.target_node_name
                    port_id = target_point.port_id

                # TODO: Use common target point class instead of tuple
                key = (_target_node_name, target_type, port_id)
                for tensor_collectors in statistic_point.algorithm_to_tensor_collectors.values():
                    target_type_to_tensor_collector_map[key]["collectors"].extend(tensor_collectors)
                target_type_to_tensor_collector_map[key]["target_point"].append(target_point)

        for merged_collectors_info in target_type_to_tensor_collector_map.values():
            target_point = merged_collectors_info["target_point"][0]
            collectors = merged_collectors_info["collectors"]
            merged_collector = MergedTensorCollector(collectors)
            stat_point = StatisticPoint(target_point, merged_collector, "Merged")
            merged_statistic_points.add_statistic_point(stat_point)
        return merged_statistic_points

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, OVNNCFTensor]:
        return {n: OVNNCFTensor(v) for n, v in outputs.items()}
