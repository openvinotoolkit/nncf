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

from nncf.common.graph.transformations.layout import TransformationLayout

from nncf.experimental.post_training.backend import BACKEND
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.graph.model_transformer import ModelTransformer
from nncf.experimental.post_training.algorithm import PostTrainingAlgorithm
from nncf.experimental.post_training.compressed_model import CompressedModel

from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithm
from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithms
from nncf.experimental.post_training.initialization.algorithm import InitizalizationParameters
from nncf.experimental.post_training.quantization.parameters import PostTrainingQuantizationParameters


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
        self.number_samples = quantization_parameters.number_samples

        self.algorithms_to_created = quantization_parameters.algorithms
        self.algorithms = []

    def apply(self, compressed_model: CompressedModel, engine: Engine) -> CompressedModel:
        model_transformer = self._create_model_transformer(compressed_model)
        transformation_layout = self._create_transformation_layout(compressed_model)
        statistics_collector = self._create_statistics_collector(compressed_model, engine)
        for algorithm, parameters in self.algorithms_to_created.items():
            algorithm = self._create_algorithm(compressed_model, engine, algorithm, parameters)
            self.algorithms.append(algorithm)

        layers_to_collect_statistics = []  # List[]
        for initialization_algorithm in self.algorithms:
            # TODO: potentially could be intersection in layers_to_collect_statistics
            layers_to_collect_statistics.extend(
                initialization_algorithm.get_layers_for_statistics(self.weight_quantizer_config,
                                                                   self.activation_quantizer_config))

        layers_statistics = statistics_collector.collect_statistics(layers_to_collect_statistics,
                                                                    self.number_samples)

        for initialization_algorithm in self.algorithms:
            transformation_commands = initialization_algorithm.get_transformation_commands(layers_statistics,
                                                                                           self.weight_quantizer_config,
                                                                                           self.activation_quantizer_config)
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

    def _create_algorithm(self, compressed_model: CompressedModel, engine: Engine, algorithm: InitializationAlgorithms,
                          parameters: InitizalizationParameters) -> InitializationAlgorithm:
        if compressed_model.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.initialization.quantizer_range_finder import ONNXQuantizerRangeFinderAlgorithm
            from nncf.experimental.onnx.initialization.batchnorm_adaptation import ONNXBatchNormAdaptationAlgorithm
            from nncf.experimental.onnx.initialization.bias_correction import ONNXBiasCorrectionAlgorithm
            if algorithm == InitializationAlgorithms.QuantizerRangeFinder:
                return ONNXQuantizerRangeFinderAlgorithm(compressed_model, engine, parameters)
            if algorithm == InitializationAlgorithms.BiasCorrection:
                return ONNXBiasCorrectionAlgorithm(compressed_model, engine, parameters)
