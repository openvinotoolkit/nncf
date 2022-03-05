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

from typing import Deque
from typing import TypeVar
from collections import deque

from nncf.common.graph.transformations.layout import TransformationLayout

from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.backend import determine_model_backend
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.algorithms import PostTrainingAlgorithms
from nncf.experimental.post_training.statistics.statistics_collector import StatisticsCollector
from nncf.experimental.post_training.algorithms.quantization.parameters import PostTrainingQuantizationParameters

ModelType = TypeVar('ModelType')


class PostTrainingQuantization(Algorithm):
    """

    1) QuantizerRangeFinder
    2) BiasCorrection

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
        self.algorithms = deque()

    def apply(self, model: ModelType, engine: Engine) -> ModelType:
        statistics_collector = self._create_statistics_collector(model, engine)
        self.algorithms = self._create_algorithms(model, statistics_collector)

        for algorithm in self.algorithms:
            layers_to_collect_statistics = algorithm.get_layers_for_statistics(model)
            statistics_collector.register_layer_statistics(layers_to_collect_statistics)

        statistics_collector.collect_statistics(model, self.number_samples)

        while len(self.algorithms) > 0:
            algorithm = self.algorithms.popleft()
            compressed_model = algorithm.apply(compressed_model, engine)

        return compressed_model

    def _create_statistics_collector(self, model: ModelType, engine: Engine) -> StatisticsCollector:
        backend = determine_model_backend(model)
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.statistics.statistics_collector import ONNXStatisticsCollector
            return ONNXStatisticsCollector(engine)

    def _create_transformation_layout(self, model: ModelType) -> TransformationLayout:
        backend = determine_model_backend(model)
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
            return ONNXTransformationLayout()

    def _create_algorithms(self, model: ModelType, statistics_collector) -> Deque[Algorithm]:
        output = deque()
        backend = determine_model_backend(model)
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.algorithms.min_max_quantization import ONNXMinMaxQuantization
            for algorithm, parameters in self.algorithms_to_created.items():
                if algorithm == PostTrainingAlgorithms.QuantizerRangeFinder:
                    output.append(ONNXMinMaxQuantization(statistics_collector, parameters))
        return output
