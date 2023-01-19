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
import openvino.runtime as ov

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.tensor import OVNNCFTensor


class OVStatisticsAggregator(StatisticsAggregator):

    def collect_statistics(self, model: ov.Model) -> None:
        self._name_to_node_mapping = {
            op.get_friendly_name(): op for op in model.get_ops()
        }
        super().collect_statistics(model)

    def _register_statistics(self,
                             outputs: Dict[str, OVNNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        for node_name, _statistic_points in statistic_points.items():
            for statistic_point in _statistic_points:
                port_id = statistic_point.target_point.port_id
                if statistic_point.target_point.type == TargetType.POST_LAYER_OPERATION:
                    stat_node_name = node_name
                elif statistic_point.target_point.type == TargetType.PRE_LAYER_OPERATION:
                    node = self._name_to_node_mapping[node_name]
                    stat_node_name = node.input_value(port_id).get_node().get_friendly_name()
                else:
                    RuntimeError('The statistics should be collected only from the inputs or outputs of the node.')

                output_name = f'Result_{stat_node_name}.{port_id}'
                statistic_point.register_tensor(outputs[output_name])

    @staticmethod
    def _get_transformation_layout_extra_outputs(statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(OVOutputInsertionCommand(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, OVNNCFTensor]:
        return {n: OVNNCFTensor(v) for n, v in outputs.items()}
