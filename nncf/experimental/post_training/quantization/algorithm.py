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
from typing import Dict
from typing import Optional

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.graph.transformations.layout import TransformationLayout

from nncf.experimental.post_training.backend import BACKEND
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.graph.model_transformer import ModelTransformer
from nncf.experimental.post_training.algorithm import PostTrainingAlgorithm
from nncf.experimental.post_training.algorithm import PostTraniningAlgorithmParameters
from nncf.experimental.post_training.compressed_model import CompressedModel

from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithm
from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithms
from nncf.experimental.post_training.initialization.algorithm import InitizalizationParameters
from nncf.experimental.post_training.initialization.quantizer_range_finder import QuantizerRangeFinderParameters

from nncf.experimental.post_training.utils import merge_two_dicts


class PostTrainingQuantizationParameters(PostTraniningAlgorithmParameters):
    def __init__(self,
                 preset: str = 'perfomance',
                 iterations_number: int = 300,
                 target_device: str = 'CPU',  # TODO: change to ENUM
                 weight_bits: int = 8,
                 activation_bits: int = 8,
                 initialization_algorithms: List[InitializationAlgorithms] = None,
                 initialization_algorithms_parameters: List[InitizalizationParameters] = None,
                 ignored_scopes: Optional[List[str]] = None
                 ):

        if initialization_algorithms is None:
            default_initialization_algorithms = [InitializationAlgorithms.QuantizerRangeFinder]
            default_initialization_algorithms_parameters = [
                QuantizerRangeFinderParameters(weight_statistics_min_func='min',
                                               weight_statistics_max_func='max',
                                               activation_statistics_min_func='min',
                                               activation_statistics_max_func='max'
                                               )]
            self.initialization_algorithms = self._determine_initialization_algorithms(
                default_initialization_algorithms,
                default_initialization_algorithms_parameters)
        else:
            # type: Dict[InitializationAlgorithms, InitizalizationParameters]
            self.initialization_algorithms = self._determine_initialization_algorithms(initialization_algorithms,
                                                                                       initialization_algorithms_parameters)
        self._determine_weight_activation_quantizers_config(
            preset,
            weight_bits,
            activation_bits)
        self.iterations_number = iterations_number
        self.target_device = target_device
        self.ignored_scopes = ignored_scopes

    def _determine_weight_activation_quantizers_config(self, preset: str, weight_bits: int, activation_bits: int):
        weight_mode = QuantizationMode.SYMMETRIC
        activation_mode = QuantizationMode.SYMMETRIC
        self.weight_quantizer_config = QuantizerConfig(num_bits=weight_bits, mode=weight_mode, per_channel=True)
        self.activation_quantizer_config = QuantizerConfig(num_bits=activation_bits, mode=activation_mode,
                                                           per_channel=False)

    def _determine_initialization_algorithms(self, initialization_algorithms, initialization_algorithms_parameters):
        output = {}
        for algorithm, parameters in zip(initialization_algorithms, initialization_algorithms_parameters):
            output[algorithm] = parameters
        return output


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

    def __init__(self,
                 quantization_parameters: PostTrainingQuantizationParameters = PostTrainingQuantizationParameters()):
        super().__init__()
        self.weight_quantizer_config = quantization_parameters.weight_quantizer_config
        self.activation_quantizer_config = quantization_parameters.activation_quantizer_config
        self.ignored_scopes = quantization_parameters.ignored_scopes
        self.target_device = quantization_parameters.target_device
        self.iterations_number = quantization_parameters.iterations_number

        self.initialization_algorithms_to_created = quantization_parameters.initialization_algorithms
        self.initialization_algorithms = []

    def apply(self, compressed_model: CompressedModel, engine: Engine) -> CompressedModel:
        model_transformer = self._create_model_transformer(compressed_model)
        transformation_layout = self._create_transformation_layout(compressed_model)
        statistics_collector = self._create_statistics_collector(compressed_model, engine)
        for algorithm, parameters in self.initialization_algorithms_to_created.items():
            algorithm = self._create_initialization_algorithm(compressed_model, engine, algorithm, parameters)
            self.initialization_algorithms.append(algorithm)

        layers_to_collect_statistics = []  # List[]
        for initialization_algorithm in self.initialization_algorithms:
            # TODO: potentially could be intersection in layers_to_collect_statistics
            layers_to_collect_statistics.extend(
                initialization_algorithm.get_layers_for_statistics(self.weight_quantizer_config,
                                                                   self.activation_quantizer_config))

        layers_statistics = statistics_collector.collect_statistics(layers_to_collect_statistics,
                                                                    self.iterations_number)

        for initialization_algorithm in self.initialization_algorithms:
            transformation_commands = initialization_algorithm.get_transformation_commands(layers_statistics)
            for transformation_command in transformation_commands:
                transformation_layout.register(transformation_command)

        model_transformer.transform(compressed_model, transformation_layout)

        return compressed_model

    def _create_model_transformer(self, compressed_model: CompressedModel) -> ModelTransformer:
        if compressed_model.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
            return ONNXModelTransformer(compressed_model)

    def _create_statistics_collector(self, compressed_model: CompressedModel, engine: Engine):
        if compressed_model.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.initialization.statistics_collector import ONNXStatisticsCollector
            return ONNXStatisticsCollector(compressed_model, engine)

    def _create_transformation_layout(self, compressed_model: CompressedModel) -> TransformationLayout:
        if compressed_model.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
            return ONNXTransformationLayout()

    def _create_initialization_algorithm(self, compressed_model: CompressedModel, engine: Engine,
                                         algorithm: InitializationAlgorithms,
                                         parameters: InitizalizationParameters) -> InitializationAlgorithm:
        if compressed_model.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.initialization.quantizer_range_finder import ONNXQuantizerRangeFinderAlgorithm
            from nncf.experimental.onnx.initialization.batchnorm_adaptation import ONNXBatchNormAdaptationAlgorithm
            from nncf.experimental.onnx.initialization.bias_correction import ONNXBiasCorrectionAlgorithm
            if algorithm == InitializationAlgorithms.QuantizerRangeFinder:
                return ONNXQuantizerRangeFinderAlgorithm(compressed_model, engine, parameters)
            if algorithm == InitializationAlgorithms.BatchNormAdaptation:
                return ONNXBatchNormAdaptationAlgorithm()
            if algorithm == InitializationAlgorithms.BiasCorrection:
                return ONNXBiasCorrectionAlgorithm()
