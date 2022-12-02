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

from nncf import Dataset
from nncf.common.utils.logger import logger as nncf_logger
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

    def __init__(self, convert_opset_version: bool = True):
        self.algorithms = []
        self.convert_opset_version = convert_opset_version

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

    def _get_prepared_model_for_compression(self, model: TModel, backend: BackendType) -> TModel:
        if backend == BackendType.ONNX:
            from nncf.experimental.onnx.model_normalizer import ONNXModelNormalizer
            return ONNXModelNormalizer.normalize_model(model, self.convert_opset_version)

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
            nncf_logger.info('There are no algorithms added. The original model will be returned.')
            return model

        backend = get_backend(model)

        # TODO (KodiaqQ): Remove after ONNX is removed from experimental
        if backend == BackendType.ONNX:
            nncf_logger.warning('You are using experimental ONNX backend for the Post-training quantization.')
        modified_model = self._get_prepared_model_for_compression(model, backend)

        statistics_aggregator = self._create_statistics_aggregator(dataset, backend)

        for algorithm in self.algorithms:
            statistic_points = algorithm.get_statistic_points(modified_model)
            statistics_aggregator.register_stastistic_points(statistic_points)

        statistics_aggregator.collect_statistics(modified_model)

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(modified_model, statistics_aggregator.statistic_points)
        return modified_model
