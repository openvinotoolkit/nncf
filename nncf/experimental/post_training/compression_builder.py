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

from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.algorithms import Algorithm

from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.backend import get_model_backend

ModelType = TypeVar('ModelType')


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

    def _create_engine(self, backend: Backend) -> Engine:
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.engine import ONNXEngine
            return ONNXEngine()
        return None

    def _create_statistics_aggregator(self, engine: Engine, dataloader: DataLoader, backend: Backend):
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.statistics.aggregator import ONNXStatisticsAggregator
            return ONNXStatisticsAggregator(engine, dataloader)
        return None

    def _get_prepared_model_for_compression(self, model: ModelType, backend: Backend) -> ModelType:
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.model_normalizer import ONNNXModelNormalizer
            return ONNNXModelNormalizer.modify_onnx_model_for_quantization(model)
        return None

    def apply(self, model: ModelType, dataloader: DataLoader, engine: Engine = None) -> ModelType:
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

        statistics_aggregator = self._create_statistics_aggregator(engine, dataloader, backend)
        for algorithm in self.algorithms:
            layers_to_collect_statistics = algorithm.get_layers_for_statistics(modified_model)
            statistics_aggregator.register_layer_statistics(layers_to_collect_statistics)

        statistics_aggregator.collect_statistics(modified_model)

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(modified_model, engine, statistics_aggregator.layers_statistics)
        return modified_model
