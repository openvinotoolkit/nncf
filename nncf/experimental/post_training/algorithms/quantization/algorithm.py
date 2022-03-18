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

from typing import Dict
from typing import TypeVar

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase

from nncf.experimental.post_training.backend import Backend
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.algorithms import PostTrainingAlgorithms
from nncf.experimental.post_training.algorithms.quantization.parameters import PostTrainingQuantizationParameters

ModelType = TypeVar('ModelType')


class PostTrainingQuantization(Algorithm):
    """
    Implements Post-Training Quantization algorithm, which basically includes:
    1) MinMaxQuantization
    2) BiasCorrection
    3) ChannelAlignment

    Disclaimer: currently, it only supports MinMaxQuantization. The following algorithms will be added soon.

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

    def _apply(self, model: ModelType, engine: Engine,
               layer_statistics: Dict[str, TensorStatisticCollectorBase]) -> ModelType:
        for algorithm in self.algorithms:
            quantized_model = algorithm.apply(model, engine, layer_statistics)
        return quantized_model

    def get_layers_for_statistics(self, model: ModelType) -> Dict[str, TensorStatisticCollectorBase]:
        output = {}
        for algorithm in self.algorithms:
            output = {**output, **algorithm.get_layers_for_statistics(model)}
        return output

    def create_subalgorithms(self, backend: Backend) -> None:
        if backend == Backend.ONNX:
            from nncf.experimental.onnx.algorithms.min_max_quantization import ONNXMinMaxQuantization
            for algorithm, parameters in self.algorithms_to_created.items():
                if algorithm == PostTrainingAlgorithms.MinMaxQuantization:
                    self.algorithms.append(ONNXMinMaxQuantization(parameters))
