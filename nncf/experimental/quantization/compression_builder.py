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

from typing import TypeVar

from copy import deepcopy

from nncf import Dataset
from nncf.common.logging import nncf_logger
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.quantization.telemetry_extractors import CompressionStartedFromBuilder

from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_ONNX_CATEGORY

TModel = TypeVar('TModel')


class CompressionBuilder:
    """
    The main class applies the compression algorithms to the model according to their order.
    """

    def __init__(self):
        self.algorithms = []

    def add_algorithm(self, algorithm: Algorithm) -> None:
        """
        Adds the algorithm to the pipeline.
        """
        self.algorithms.append(algorithm)

    def _create_statistics_aggregator(self,
                                      dataset: Dataset,
                                      backend: BackendType) -> StatisticsAggregator:
        """
        Creates backend-specific StatisticsAggregator.

        :param engine: engine for the model execution
        :param dataset: dataset for the statistics collection and validation
        :param model_transformer: backend-specific ModelTransformerBase instance
        :param backend: model backend type for the further differentiations
        :return: backnd-specific StatisticsAggregator
        """
        if backend == BackendType.ONNX:
            from nncf.experimental.onnx.statistics.aggregator import \
                ONNXStatisticsAggregator
            return ONNXStatisticsAggregator(dataset)
        return None

    @tracked_function(NNCF_ONNX_CATEGORY, [CompressionStartedFromBuilder(argname="self"), ])
    def apply(self, model: TModel, dataset: Dataset) -> TModel:
        """
        Apply compression algorithms to the 'model'.

        1) Prepare the original model. This step is essential for some backends, e.g. ONNX
        2) Creates subalgorithms, which is essential for some composite algorithms such as PostTrainingQuantization
        2) Creates default Engine if it wasn't provided.
        3) Creates StatisticsAggregator.
        4) Get layers for statistics collection from algorithms.
        5) Collect all statistics.
        6) Apply algorithms.
        """
        if not self.algorithms:
            nncf_logger.warning('No algorithms specified for compression - '
                                'doing nothing and returning the original model')
            return model

        _model = deepcopy(model)
        backend = get_backend(_model)

        # TODO (KodiaqQ): Remove after ONNX is removed from experimental
        if backend == BackendType.ONNX:
            nncf_logger.warning('You are using the experimental ONNX backend for post-training quantization.')

        statistics_aggregator = self._create_statistics_aggregator(dataset, backend)

        for algorithm in self.algorithms:
            statistic_points = algorithm.get_statistic_points(_model)
            statistics_aggregator.register_stastistic_points(statistic_points)

        statistics_aggregator.collect_statistics(_model)

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(_model, statistics_aggregator.statistic_points)
        return modified_model
