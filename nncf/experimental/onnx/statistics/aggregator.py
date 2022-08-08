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

import onnx

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.post_training.api.dataset import Dataset
from nncf.experimental.post_training.statistics.aggregator import StatisticsAggregator
from nncf.experimental.post_training.api.sampler import Sampler
from nncf.experimental.onnx.samplers import ONNXBatchSampler
from nncf.experimental.onnx.samplers import ONNXRandomBatchSampler
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.graph.transformations.commands import ONNXOutputInsertionCommand


class ONNXStatisticsAggregator(StatisticsAggregator):
    # TODO (Nikita Malinin): Remove ONNXStatisticsAggregator & create the common backend-agnostic solution
    def __init__(self, engine: ONNXEngine, dataset: Dataset):
        super().__init__(engine, dataset)

    def _get_transformation_layout_extra_outputs(self, model):
        transformation_layout = ONNXTransformationLayout()
        transformation_commands = []
        for _statistic_points in self.statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(ONNXOutputInsertionCommand(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    def _create_model_transformer(self, model: onnx.ModelProto) -> ONNXModelTransformer:
        return ONNXModelTransformer(model)

    def _create_sampler(self, dataset: Dataset,
                        sample_indices: int) -> Sampler:
        if dataset.shuffle:
            nncf_logger.info('Using Shuffled dataset')
            return ONNXRandomBatchSampler(dataset, sample_indices=sample_indices)
        nncf_logger.info('Using Non-Shuffled dataset')
        return ONNXBatchSampler(dataset, sample_indices=sample_indices)
