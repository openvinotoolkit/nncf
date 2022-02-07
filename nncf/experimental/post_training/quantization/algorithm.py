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

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.graph.transformations.layout import TransformationLayout

from nncf.experimental.post_training.backend import define_the_backend
from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.api.engine import Engine
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
                 transformation_algorithms: None,
                 initialization_algorithms: List[Type[InitializationAlgorithm]]
                 ):
        mode = QuantizationMode.SYMMETRIC if symmetric_weights else QuantizationMode.ASYMMETRIC
        self.weight_quantizer_config = QuantizerConfig(num_bits=8,
                                                       mode=mode,
                                                       per_channel=per_channel_weights)
        mode = QuantizationMode.SYMMETRIC if symmetric_activations else QuantizationMode.ASYMMETRIC
        self.activation_quantizer_config = QuantizerConfig(num_bits=8,
                                                           mode=mode,
                                                           per_channel=False)

        self.ignored_scopes = ignored_scopes
        self.quantize_inputs = quantize_inputs
        self.quantize_outputs = quantize_outputs
        self.target_device = target_device
        self.transformation_algorithms = transformation_algorithms
        self.initialization_algorithms = initialization_algorithms


DEFAULT = PostTrainingQuantizationParameters(per_channel_weights=True,
                                             symmetric_weights=True,
                                             symmetric_activations=True,
                                             ignored_scopes=None,
                                             quantize_inputs=True,
                                             quantize_outputs=True,
                                             target_device='CPU',
                                             transformation_algorithms=None,
                                             initialization_algorithms=[QuantizerRangeFinderAlgorithm])


class InitializationAlgorithmPriority(OrderedEnum):
    DEFAULT_PRIORITY = 1
    QUANTIZER_RANGE_FIND = 2
    BATCH_NORM_ADAPTATION = 3
    BIAS_CORRECTION = 4


def merge_two_dicts(x, y):
    # TODO: check intersections keys
    z = x.copy()
    z.update(y)
    return z


class PostTrainingQuantization(PostTrainingAlgorithm):
    """
    Post-Training Quantization algorithm makes 3 main things:
        1) Find the transformations needed to apply to the model.
        2) Apply these transformations.
        3) Initialize the transformed model.
    """

    def __init__(self, quantization_parameters: PostTrainingQuantizationParameters):
        super().__init__()
        self.weight_quantizer_config = quantization_parameters.weight_quantizer_config
        self.activation_quantizer_config = quantization_parameters.activation_quantizer_config
        self.ignored_scopes = quantization_parameters.ignored_scopes
        self.quantize_inputs = quantization_parameters.quantize_inputs
        self.quantize_outputs = quantization_parameters.quantize_outputs
        self.target_device = quantization_parameters.target_device

        self.transformation_algorithms_to_created = quantization_parameters.transformation_algorithms
        self.transformation_algorithms = []

        self.initialization_algorithms_to_created = quantization_parameters.initialization_algorithms
        self.initialization_algorithms = []

    def apply(self, compressed_model: CompressedModel, engine: Engine) -> CompressedModel:
        model_transformer = self._create_model_transformer(compressed_model)
        transformation_layout = self._create_transformation_layout(compressed_model)
        statistics_collector = self._create_statistics_collector(compressed_model, engine)
        for algorithm in self.initialization_algorithms_to_created:
            algorithm = self._create_initialization_algorithm(compressed_model, engine, algorithm)
            self.initialization_algorithms.append(algorithm)

        # Algorithms to prepare FP32 model for initialiation algorithms, e.g. ChannelAlignment
        for transformation_algorithm in self.transformation_algorithms:
            transformation_algorithm.run(compressed_model)

        layers_to_collect_statistics = {}  # Dict[str: Callable]
        for initialization_algorithm in self.initialization_algorithms:
            layers_to_collect_statistics = merge_two_dicts(layers_to_collect_statistics,
                                                           initialization_algorithm.get_layers_for_statistics())

        statistics_collector.collect_statistics(layers_to_collect_statistics, 10)

        for initialization_algorithm in self.initialization_algorithms:
            transformation_commands = initialization_algorithm.get_transformation_commands(statistics_collector)
            for transformation_command in transformation_commands:
                transformation_layout.register(transformation_command)

        model_transformer.transform(compressed_model, transformation_layout)

        return compressed_model

    def _create_model_transformer(self, compressed_model: CompressedModel) -> ModelTransformer:
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
            return ONNXModelTransformer(compressed_model)

    def _create_statistics_collector(self, compressed_model: CompressedModel, engine: Engine):
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            from nncf.experimental.onnx.initialization.statistics_collector import ONNXStatisticsCollector
            return ONNXStatisticsCollector(compressed_model, engine)

    def _create_transformation_layout(self, compressed_model: CompressedModel) -> TransformationLayout:
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
            return ONNXTransformationLayout()

    def _create_initialization_algorithm(self, compressed_model: CompressedModel, engine: Engine,
                                         algorithm: Type[InitializationAlgorithm]) -> InitializationAlgorithm:
        if define_the_backend(compressed_model.original_model) == Backend.ONNX:
            if algorithm is QuantizerRangeFinderAlgorithm:
                return ONNXQuantizerRangeFinderAlgorithm(compressed_model, engine)
            if algorithm is BatchNormAdaptationAlgorithm:
                return ONNXBatchNormAdaptationAlgorithm()
            if algorithm is BiasCorrectionAlgorithm:
                return ONNXBiasCorrectionAlgorithm()
