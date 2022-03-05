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

    def apply(self, model: ModelType, dataloader: DataLoader, engine: Engine = None) -> ModelType:
        """
        Apply compression algorithms to the 'model'.
        """
        if not self.algorithms:
            print('There are no algorithms added. The original model will be returned.')
            return model
        if engine is None:
            engine = self._create_engine(model, dataloader)
        while self.algorithms:
            algorithm = self.algorithms.pop()
            compressed_model = algorithm.apply(model, engine)
        return compressed_model
