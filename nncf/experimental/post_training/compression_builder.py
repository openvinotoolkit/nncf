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

from collections import deque

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.algorithms import Algorithm

from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.backend import determine_model_backend

ModelType = TypeVar('ModelType')


class CompressionBuilder:
    """
    The main class applies the compression algorithms to the model according to their order.
    """

    def __init__(self):
        self.algorithms = deque()

    def add_algorithm(self, algorithm: Algorithm) -> None:
        """
        Adds the algorithm to the pipeline.
        """
        self.algorithms.append(algorithm)

    def _create_engine(self, model: ModelType, dataloader: DataLoader) -> Engine:
        backend = determine_model_backend(model)
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.engine import ONNXEngine
            return ONNXEngine(dataloader)
        return None

    def _create_statistics_collector(self, model: ModelType, engine: Engine):
        backend = determine_model_backend(model)
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.statistics.statistics_collector import ONNXStatisticsCollector
            return ONNXStatisticsCollector(engine)
        return None

    def _get_prepared_model_for_compression(self, model: ModelType) -> ModelType:
        backend = determine_model_backend(model)
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.helper import modify_onnx_model_for_quantization
            return modify_onnx_model_for_quantization(model)

    def apply(self, model: ModelType, dataloader: DataLoader, engine: Engine = None) -> ModelType:
        """
        Apply compression algorithms to the 'model'.
        """

        if not self.algorithms:
            print('There are no algorithms added. The original model will be returned.')
            return model

        modified_model = self._get_prepared_model_for_compression(model)

        if engine is None:
            engine = self._create_engine(modified_model, dataloader)

        statistics_collector = self._create_statistics_collector(modified_model, engine)
        for algorithm in self.algorithms:
            layers_to_collect_statistics = algorithm.get_layers_for_statistics(modified_model)
            statistics_collector.register_layer_statistics(layers_to_collect_statistics, None)

        statistics_collector.collect_statistics(modified_model)

        while self.algorithms:
            algorithm = self.algorithms.pop()
            modified_model = algorithm.apply(modified_model, engine, statistics_collector.layers_statistics)
        return modified_model
