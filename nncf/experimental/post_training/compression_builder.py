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

from typing import Callable, Optional, TypeVar

from copy import deepcopy

from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.post_training.algorithms.algorithm import CompositeAlgorithm

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.metric import Metric
from nncf.experimental.post_training.api.dataset import Dataset
from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.statistics.aggregator import StatisticsAggregator

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

    def _create_engine(self, backend: BackendType) -> Engine:
        """
        Creates backend-specific Engine.

        :param backend: model backend type for the further differentiations
        :return: backnd-specific Engine
        """
        if backend == BackendType.ONNX:
            from nncf.experimental.onnx.engine import ONNXEngine
            return ONNXEngine()
        return None

    def _create_statistics_aggregator(self,
                                      engine: Engine,
                                      dataset: Dataset,
                                      backend: BackendType) -> StatisticsAggregator:
        """
        Creates backend-specific StatisticsAggregator.

        :param engine: engine for the model execution
        :param dataset: dataset for the statistics collection and validation
        :param model_transformer: backend-specific StaticModelTransformerBase instance
        :param backend: model backend type for the further differentiations
        :return: backnd-specific StatisticsAggregator
        """
        if backend == BackendType.ONNX:
            from nncf.experimental.onnx.statistics.aggregator import \
                ONNXStatisticsAggregator
            return ONNXStatisticsAggregator(engine, dataset)
        return None

    def _create_model_transformer(self, model: TModel, backend: BackendType) -> ModelTransformer:
        """
        Creates backend-specific ModelTransformer.

        :param model: input model for the ModelTransformer
        :param backend: model backend type for the further differentiations
        :return: backnd-specific ModelTransformer
        """
        if backend == BackendType.ONNX:
            from nncf.experimental.onnx.graph.model_transformer import \
                ONNXModelTransformer
            return ONNXModelTransformer(model)
        return None

    def apply(self, model: TModel, dataset: Dataset, engine: Engine = None) -> TModel:
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
        _model = deepcopy(model)
        backend = get_backend(_model)

        if engine is None:
            engine = self._create_engine(backend)

        for algorithm in self.algorithms:
            if isinstance(algorithm, CompositeAlgorithm):
                algorithm.create_subalgorithms()

        statistics_aggregator = self._create_statistics_aggregator(engine, dataset, backend)
        for algorithm in self.algorithms:
            statistic_points = algorithm.get_statistic_points(_model)
            statistics_aggregator.register_stastistic_points(statistic_points)

        model_transformer = self._create_model_transformer(_model, backend)
        statistics_aggregator.collect_statistics(model_transformer)

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(_model, engine, statistics_aggregator.statistic_points)
        return modified_model

    def evaluate(self, model: TModel, metric: Metric, dataset: Dataset,
                 engine: Engine = None, outputs_transforms: Optional[Callable] = None):
        backend = get_backend(model)

        if engine is None:
            engine = self._create_engine(backend)
        if outputs_transforms is not None:
            engine.set_outputs_transforms(outputs_transforms)
        engine.set_model(model)
        engine.set_metrics(metric)
        engine.set_dataset(dataset)
        return engine.compute_metrics()
