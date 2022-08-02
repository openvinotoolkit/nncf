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

from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.metric import Metric
from nncf.experimental.post_training.api.dataset import Dataset
from nncf.experimental.post_training.algorithms import Algorithm

from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.backend import get_model_backend

ModelType = TypeVar('ModelType')


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

    def _create_engine(self, backend: Backend) -> Engine:
        # TODO (Nikita Malinin): Place "ifs" into the backend-specific expandable structure
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.engine import ONNXEngine  # pylint: disable=cyclic-import
            return ONNXEngine()
        return None

    def _create_statistics_aggregator(self, engine: Engine, dataset: Dataset, backend: Backend):
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.statistics.aggregator import \
                ONNXStatisticsAggregator  # pylint: disable=cyclic-import
            return ONNXStatisticsAggregator(engine, dataset)
        return None

    def _get_prepared_model_for_compression(self, model: ModelType, backend: Backend) -> ModelType:
        # TODO (Nikita Malinin): Replace this methood into backend-specific graph transformer
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.model_normalizer import ONNXModelNormalizer  # pylint: disable=cyclic-import
            if self.convert_opset_version:
                model = ONNXModelNormalizer.convert_opset_version(model)
            return ONNXModelNormalizer.replace_empty_node_name(model)

        return None

    def apply(self, model: ModelType, dataset: Dataset, engine: Engine = None) -> ModelType:
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

        backend = get_model_backend(model)
        modified_model = self._get_prepared_model_for_compression(model, backend)

        if engine is None:
            engine = self._create_engine(backend)

        for algorithm in self.algorithms:
            algorithm.create_subalgorithms(backend)

        statistics_aggregator = self._create_statistics_aggregator(engine, dataset, backend)
        for algorithm in self.algorithms:
            statistic_points = algorithm.get_statistic_points(modified_model)
            statistics_aggregator.register_stastistic_points(statistic_points)

        statistics_aggregator.collect_statistics(modified_model)

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(modified_model, engine, statistics_aggregator.statistic_points)
        return modified_model

    def evaluate(self, model: ModelType, metric: Metric, dataset: Dataset,
                 engine: Engine = None, outputs_transforms: Optional[Callable] = None):
        backend = get_model_backend(model)

        if engine is None:
            engine = self._create_engine(backend)
        if outputs_transforms is not None:
            engine.set_outputs_transforms(outputs_transforms)
        engine.set_model(model)
        engine.set_metrics(metric)
        engine.set_dataset(dataset)
        return engine.compute_metrics()
