"""
 Copyright (c) 2022 Intel Corporation
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

from nncf.common.graph.transformations.layout import TransformationLayout
from nncf import Dataset
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.graph.transformations.commands import ONNXOutputInsertionCommand


class ONNXStatisticsAggregator(StatisticsAggregator):
    def __init__(self, engine: ONNXEngine, dataset: Dataset):
        super().__init__(engine, dataset)

    def _get_transformation_layout_extra_outputs(
            self,
            statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(ONNXOutputInsertionCommand(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout
