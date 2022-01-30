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
from nncf.experimental.post_training_api.compressed_model import CompressedModel
from nncf.common.compression import BaseCompressionAlgorithmBuilder

ModelType = TypeVar('ModelType')


class AlgorithmPriority(OrderedEnum):
    DEFAULT_PRIORITY = 0
    PRUNING_PRIORITY = 2
    SPARSIFICATION_PRIORITY = 3
    QUANTIZATION_PRIORITY = 11


class CompressionBuilder:
    """
    The main class holds the compression algorithms and
    controls the compression algorithms flow applied to CompressedModel.
    """

    def __init__(self):
        self.algorithms = PriorityQueue()

    def add_algorithm(self, algorithm: BaseCompressionAlgorithmBuilder,
                      priority: AlgorithmPriority = None) -> None:
        if priority is not None:
            self._set_algorithm_priority(algorithm, priority)
        else:
            priority = self._define_algorithm_priority(algorithm)
            self._set_algorithm_priority(algorithm, priority)
        self.algorithms.add(algorithm)

    def _set_algorithm_priority(self, algorithm: BaseCompressionAlgorithmBuilder, priority: AlgorithmPriority) -> None:
        algorithm.priority = AlgorithmPriority.QUANTIZATION_PRIORITY

    def _define_algorithm_priority(self, algorithm: BaseCompressionAlgorithmBuilder) -> AlgorithmPriority:
        # TODO: need to realize
        return AlgorithmPriority.DEFAULT_PRIORITY

    def init(self, model: ModelType) -> CompressedModel:
        """
        Apply compression algorithms to CompressedModel according to algorithms priority.
        """
        compressed_model = CompressedModel(model)
        while not self.algorithms.is_empty():
            algorithm = self.algorithms.pop()
            algorithm.apply_to(compressed_model)
        return compressed_model
