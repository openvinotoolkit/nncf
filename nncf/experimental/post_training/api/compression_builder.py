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

from nncf.common.utils.priority_queue import PriorityQueue
from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.algorithm import PostTrainingAlgorithm

from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantization
from nncf.experimental.post_training.sparsity.algorithm import PostTrainingSpasity

from nncf.experimental.post_training.backend import define_the_backend
from nncf.experimental.post_training.backend import Backend

ModelType = TypeVar('ModelType')


class CompressionAlgorithmPriority(OrderedEnum):
    DEFAULT_PRIORITY = 1
    PRUNING_PRIORITY = 2
    SPARSITY_PRIORITY = 3
    QUANTIZATION_PRIORITY = 4


class CompressionBuilder:
    """
    The main class applies the compression algorithms to the model according to their order.
    """

    def __init__(self):
        self.algorithms = PriorityQueue()

    def add_algorithm(self, algorithm: PostTrainingAlgorithm,
                      priority: CompressionAlgorithmPriority = None) -> None:
        if priority is not None:
            self._set_algorithm_priority(algorithm, priority)
        else:
            priority = self._define_algorithm_priority(algorithm)
            self._set_algorithm_priority(algorithm, priority)
        self.algorithms.add(algorithm)

    def _set_algorithm_priority(self, algorithm: PostTrainingAlgorithm,
                                priority: CompressionAlgorithmPriority) -> None:
        algorithm.priority = priority

    def _define_algorithm_priority(self, algorithm: PostTrainingAlgorithm) -> CompressionAlgorithmPriority:
        """
        Defines the priority of the algorithm based on its instance.
        """
        if isinstance(algorithm, PostTrainingQuantization):
            return CompressionAlgorithmPriority.QUANTIZATION_PRIORITY
        if isinstance(algorithm, PostTrainingSpasity):
            return CompressionAlgorithmPriority.SPARSITY_PRIORITY

    def _create_compressed_model(self, model: ModelType, dataloader: DataLoader, engine: Engine) -> CompressedModel:
        """
        Creates backend-specific CompressedModel instance based on the model.
        """
        if define_the_backend(model) == Backend.ONNX:
            from nncf.experimental.onnx.compressed_model import ONNXCompressedModel
            return ONNXCompressedModel(model, dataloader, engine)
        elif define_the_backend(model) == Backend.PYTORCH:
            pass
        elif define_the_backend(model) == Backend.TENSORFLOW:
            pass
        elif define_the_backend(model) == Backend.OPENVINO:
            pass

    def _create_engine(self, model: ModelType) -> Engine:
        if define_the_backend(model) == Backend.ONNX:
            from nncf.experimental.onnx.engine import ONNXEngine
            return ONNXEngine()
        elif define_the_backend(model) == Backend.PYTORCH:
            pass
        elif define_the_backend(model) == Backend.TENSORFLOW:
            pass
        elif define_the_backend(model) == Backend.OPENVINO:
            pass

    def apply(self, model: ModelType, dataloader: DataLoader, engine: Engine = None) -> CompressedModel:
        """
        Apply compression algorithms to the 'model'.
        """
        if engine is None:
            engine = self._create_engine(model)
        compressed_model = self._create_compressed_model(model, dataloader, engine)
        while not self.algorithms.is_empty():
            algorithm = self.algorithms.pop()
            compressed_model = algorithm.apply(compressed_model, dataloader, engine)
        return compressed_model
