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
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator

from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.graph.nncf_graph_builder import get_operation_const_op
from nncf.experimental.openvino_native.tensor import OVNNCFTensor


class OVStatisticsAggregator(StatisticsAggregator):

    def collect_statistics(self, model: ov.Model) -> None:
        self._name_to_node_mapping = {
            op.get_friendly_name(): op for op in model.get_ops()
        }
        self._registered_weights = set()
        super().collect_statistics(model)

    def _register_activation_statistic(self, statistic_point: StatisticPointsContainer,
                                       target_point: OVTargetPoint,
                                       node_name: str,
                                       outputs: Dict[str, OVNNCFTensor]) -> None:
        port_id = target_point.port_id
        if target_point.type == TargetType.POST_LAYER_OPERATION:
            stat_node_name = node_name
        elif target_point.type == TargetType.PRE_LAYER_OPERATION:
            node = self._name_to_node_mapping[node_name]
            stat_node_name = node.input_value(port_id).get_node().get_friendly_name()
        output_name = f'Result_{stat_node_name}.{port_id}'
        statistic_point.register_tensor(outputs[output_name])

    def _register_weight_statistic(self, statistic_point: StatisticPointsContainer,
                                   target_point: OVTargetPoint) -> None:
        op = self._name_to_node_mapping[target_point.target_node_name]
        weight_node = get_operation_const_op(op, target_point.port_id)
        # TODO(l-bat): Add Result for weight nodes.
        weight_tensor = weight_node.get_vector().reshape(weight_node.get_output_shape(0))
        statistic_point.register_tensor(OVNNCFTensor(weight_tensor))


    def _register_statistics(self,
                             outputs: Dict[str, OVNNCFTensor],
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
                    RuntimeError(f'Unsupported target point type for statistic aggregator:'
                                 f' {target_point.type}')

    def _get_transformation_layout_extra_outputs(self,
                                                 statistic_points: StatisticPointsContainer) -> TransformationLayout:
        def is_activation_point(statistic_point: StatisticPoint) -> bool:
            return not statistic_point.target_point.is_weight_target_point()

        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                if is_activation_point(_statistic_point):
                    transformation_commands.append(OVOutputInsertionCommand(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, OVNNCFTensor]:
        return {n: OVNNCFTensor(v) for n, v in outputs.items()}
