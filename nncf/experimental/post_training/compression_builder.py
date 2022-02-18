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

from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.algorithms import Algorithm

from nncf.experimental.post_training.backend import BACKEND

ModelType = TypeVar('ModelType')


class CompressionBuilder:
    """
    The main class applies the compression algorithms to the model according to their order.
    """

    def __init__(self):
        self.algorithms = deque()

    def add_algorithm(self, algorithm: Algorithm) -> None:
        self.algorithms.append(algorithm)

    def _create_engine(self, compressed_model: CompressedModel, dataloader: DataLoader) -> Engine:
        if compressed_model.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.engine import ONNXEngine
            return ONNXEngine(dataloader)
        elif compressed_model.model_backend == BACKEND.PYTORCH:
            pass
        elif compressed_model.model_backend == BACKEND.TENSORFLOW:
            pass
        elif compressed_model.model_backend == BACKEND.OPENVINO:
            pass

    def _create_compressed_model(self, model: ModelType) -> CompressedModel:
        return CompressedModel(model)

    def apply(self, model: ModelType, dataloader: DataLoader,
              engine: Engine = None) -> CompressedModel:
        """
        Apply compression algorithms to the 'model'.
        """
        compressed_model = self._create_compressed_model(model)
        if engine is None:
            engine = self._create_engine(compressed_model, dataloader)
        compressed_model.build_and_set_nncf_graph(dataloader, engine)
        while len(self.algorithms) > 0:
            algorithm = self.algorithms.pop()  # TODO: will remove the last element. Is it expected behavior?
            compressed_model = algorithm.apply(compressed_model, engine)
        return compressed_model.compressed_model
