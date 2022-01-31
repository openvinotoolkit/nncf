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

from typing import List
from typing import Type

from nncf.common.utils.priority_queue import PriorityQueue
from nncf.common.utils.ordered_enum import OrderedEnum

from nncf.experimental.post_training.backend import define_the_backend
from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.graph.model_analyzer import ModelAnalyzer
from nncf.experimental.post_training.graph.model_transformer import ModelTransformer
from nncf.experimental.post_training.algorithm import PostTrainingAlgorithm
from nncf.experimental.post_training.compressed_model import CompressedModel

from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithm
from nncf.experimental.post_training.initialization.quantizer_range_finder import QuantizerRangeFinderAlgorithm
from nncf.experimental.post_training.initialization.batchnorm_adaptation import BatchNormAdaptationAlgorithm
from nncf.experimental.post_training.initialization.bias_correction import BiasCorrectionAlgorithm

from nncf.experimental.onnx.initialization.quantizer_range_finder import ONNXQuantizerRangeFinderAlgorithm
from nncf.experimental.onnx.initialization.batchnorm_adaptation import ONNXBatchNormAdaptationAlgorithm
from nncf.experimental.onnx.initialization.bias_correction import ONNXBiasCorrectionAlgorithm


class PostTrainingQuantizationParameters:
    def __init__(self, per_channel_weights: bool,
                 symmetric_weights: bool,
                 symmetric_activations: bool,
                 ignored_scopes: List[str],
                 quantize_inputs: bool,
                 quantize_outputs: bool,
                 target_device: str,  # TODO: change to ENUM
                 initialization_algorithms: List[Type[InitializationAlgorithm]]
                 ):
        self.per_channel_weights = per_channel_weights
        self.symmetric_weights = symmetric_weights
        self.symmetric_activations = symmetric_activations
        self.ignored_scopes = ignored_scopes
        self.quantize_inputs = quantize_inputs
        self.quantize_outputs = quantize_outputs
        self.target_device = target_device
        self.initialization_algorithms = initialization_algorithms


DEFAULT = PostTrainingQuantizationParameters(per_channel_weights=True,
                                             symmetric_weights=True,
                                             symmetric_activations=True,
                                             ignored_scopes=None,
                                             quantize_inputs=True,
                                             quantize_outputs=True,
                                             initialization_algorithms=[QuantizerRangeFinderAlgorithm,
                                                                        BiasCorrectionAlgorithm,
                                                                        BatchNormAdaptationAlgorithm])


class InitializationAlgorithmPriority(OrderedEnum):
    DEFAULT_PRIORITY = 1
    QUANTIZER_RANGE_FIND = 2
    BATCH_NORM_ADAPTATION = 3
    BIAS_CORRECTION = 4


class PostTrainingQuantization(PostTrainingAlgorithm):
    """
    Post-Training Quantization algorithm makes 3 main things:
        1) Find the transformations needed to apply to the model.
        2) Apply these transformations.
        3) Initialize the transformed model.
    """

    def __init__(self, quantization_parameters: PostTrainingQuantizationParameters):
        super().__init__()
        self.per_channel_weights = quantization_parameters.per_channel_weights
        self.symmetric_weights = quantization_parameters.symmetric_weights
        self.symmetric_activations = quantization_parameters.symmetric_activations
        self.ignored_scopes = quantization_parameters.ignored_scopes
        self.quantize_inputs = quantization_parameters.quantize_inputs
        self.quantize_outputs = quantization_parameters.quantize_outputs
        self.target_device = quantization_parameters.target_device

        self.initialization_algorithms_to_created = quantization_parameters.initialization_algorithms
        self.initialization_algorithms = PriorityQueue()

    def apply(self, compressed_model: CompressedModel, dataloader: DataLoader, engine: Engine) -> CompressedModel:
        model_analyzer = self._create_model_analyzer(compressed_model)
        model_transformer = self._create_model_transformer(compressed_model)

        for algorithm in self.initialization_algorithms_to_created:
            algorithm = self._create_initialization_algorithm(compressed_model, algorithm)
            priority = self._define_algorithm_priority(algorithm)
            self._set_algorithm_priority(algorithm, priority)
            self.initialization_algorithms.add(algorithm)

        quantization_transformations = model_analyzer.get_quantization_transformations(compressed_model)
        transformed_compressed_model = model_transformer.transform(compressed_model, quantization_transformations)

        while not self.initialization_algorithms.is_empty():
            initialization_algorithm = self.initialization_algorithms.pop()
            initialized_compressed_model = initialization_algorithm.apply(transformed_compressed_model)

        return initialized_compressed_model

    def _create_model_analyzer(self, compressed_model: CompressedModel) -> ModelAnalyzer:
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            from nncf.experimental.onnx.graph.model_analyzer import ONNXModelAnalyzer
            return ONNXModelAnalyzer()

    def _create_model_transformer(self, compressed_model: CompressedModel) -> ModelTransformer:
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
            return ONNXModelTransformer()

    def _create_initialization_algorithm(self, compressed_model: CompressedModel,
                                         algorithm: Type[InitializationAlgorithm]) -> InitializationAlgorithm:
        """
        Creates backend-specific CompressedModel instance based on the model.
        """
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            if isinstance(algorithm, QuantizerRangeFinderAlgorithm):
                return ONNXQuantizerRangeFinderAlgorithm()
            if isinstance(algorithm, BatchNormAdaptationAlgorithm):
                return ONNXBatchNormAdaptationAlgorithm()
            if isinstance(algorithm, BiasCorrectionAlgorithm):
                return ONNXBiasCorrectionAlgorithm()

    def _set_algorithm_priority(self, algorithm: InitializationAlgorithm,
                                priority: InitializationAlgorithmPriority) -> None:
        algorithm.priority = priority

    def _define_algorithm_priority(self, algorithm: InitializationAlgorithm) -> InitializationAlgorithmPriority:
        """
        Defines the priority of the algorithm based on its instance.
        """
        if isinstance(algorithm, QuantizerRangeFinderAlgorithm):
            return InitializationAlgorithmPriority.QUANTIZER_RANGE_FIND
        if isinstance(algorithm, BatchNormAdaptationAlgorithm):
            return InitializationAlgorithmPriority.BATCH_NORM_ADAPTATION
        if isinstance(algorithm, BiasCorrectionAlgorithm):
            return InitializationAlgorithmPriority.BIAS_CORRECTION

# Не хватает объяснения нужности новых сущностей (бенефиты от них)
#
