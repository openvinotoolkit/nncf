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

from typing import Dict, List

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.collectors import MergedTensorCollector
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator

from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.experimental.openvino_native.graph.node_utils import get_result_node_name

from nncf.experimental.openvino_native.graph.transformations.commands import OVInplaceStatisticInsertionCommand
from nncf.experimental.openvino_native.graph.nncf_graph_builder import get_operation_const_op
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.data.dataset import Dataset
from nncf.experimental.openvino_native.graph.node_utils import get_result_node_name

class OVStatisticsAggregator(StatisticsAggregator):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._spec_points: StatisticPointsContainer = None

    def collect_statistics(self, model: ov.Model) -> None:
        self._name_to_node_mapping = {
            op.get_friendly_name(): op for op in model.get_ops()
        }
        super().collect_statistics(model)

    def _register_statistics(self,
                             outputs: Dict[str, OVNNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            target_point = statistic_point.target_point
            node_name = target_point.target_node_name
            port_id = target_point.port_id
            if target_point.type == TargetType.POST_LAYER_OPERATION:
                stat_node_name = node_name
            elif target_point.type in [TargetType.PRE_LAYER_OPERATION,
                                       TargetType.OPERATION_WITH_WEIGHTS]:
                node = self._name_to_node_mapping[node_name]
                stat_node_name = node.input_value(port_id).get_node().get_friendly_name()
            else:
                RuntimeError(f'Unsupported target point type for statistic aggregator:'
                             f' {target_point.type}')

            tensor_collector.register_input(stat_node_name, port_id, outputs)

    def register_stastistic_points(self, statistic_points: StatisticPointsContainer) -> None:
        for target_node_name in statistic_points.data:
            for statistic_point in\
                statistic_points.iter_through_statistic_points_in_target_node(target_node_name,
                                                                              lambda x: True):
                tensor_collectors_for_all_algos = []
                for algorithm, tensor_collectors in statistic_point.algorithm_to_tensor_collectors.items():
                    tensor_collectors_for_all_algos.extend(tensor_collectors)

                merged_collector = MergedTensorCollector(tensor_collectors_for_all_algos)
                stat_point = StatisticPoint(statistic_point.target_point, merged_collector, 'Merged')
                self.merged_statistic_points.add_statistic_point(stat_point)
        super().register_stastistic_points(statistic_points)

    def _get_transformation_layout_extra_outputs(self,
                                                 statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            for op_fn in tensor_collector.get_inplace_fn():
                transformation_commands.append(
                    OVInplaceStatisticInsertionCommand(statistic_point.target_point, op_fn))
            if tensor_collector.any_stat_out_of_place():
                transformation_commands.append(OVOutputInsertionCommand(statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, OVNNCFTensor]:
        return {n: OVNNCFTensor(v) for n, v in outputs.items()}
