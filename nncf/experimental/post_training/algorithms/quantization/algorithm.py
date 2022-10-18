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
from typing import List
from typing import Union
from typing import Optional
from typing import TypeVar

from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.structs import QuantizationPreset

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.algorithms import CompositeAlgorithm
from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms import PostTrainingAlgorithms
from nncf.experimental.post_training.algorithms.quantization.min_max.algorithm import MinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization.min_max.algorithm import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.quantization.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.experimental.post_training.algorithms.quantization.fast_bias_correction.algorithm import \
    FastBiasCorrectionParameters
from nncf.experimental.post_training.algorithms.quantization.definitions import Granularity
from nncf.experimental.post_training.algorithms.quantization.definitions import RangeType
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')


class PostTrainingQuantizationParameters(AlgorithmParameters):
    """
    This class handles parameters for PostTrainingQuantization algorithm.
    """

    def __init__(self,
                 preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                 weight_bits: int = 8,
                 weight_granularity: Granularity = Granularity.PERCHANNEL,
                 activation_bits: int = 8,
                 activation_granularity: Granularity = Granularity.PERTENSOR,
                 range_type: RangeType = RangeType.MEAN_MINMAX,
                 number_samples: int = 300,
                 target_device: HWConfigType = HWConfigType.CPU,
                 quantize_outputs: bool = False,
                 ignored_scopes: Optional[List[str]] = None
                 ):
        self.algorithms = {PostTrainingAlgorithms.MinMaxQuantization: MinMaxQuantizationParameters(
            preset=preset,
            weight_bits=weight_bits,
            weight_granularity=weight_granularity,
            activation_bits=activation_bits,
            activation_granularity=activation_granularity,
            range_type=range_type,
            number_samples=number_samples,
            target_device=target_device,
            quantize_outputs=quantize_outputs,
            ignored_scopes=ignored_scopes
        ),
        PostTrainingAlgorithms.FastBiasCorrection: FastBiasCorrectionParameters(
            number_samples=number_samples
        )}

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        pass


class PostTrainingQuantization(CompositeAlgorithm):
    """
    Implements Post-Training Quantization algorithm, which basically includes:
    1) MinMaxQuantization
    2) FastBiasCorrection
    3) ChannelAlignment

    Disclaimer: currently, it only supports MinMaxQuantization. The following algorithms will be added soon.

    """

    def __init__(self,
                 quantization_parameters: PostTrainingQuantizationParameters = PostTrainingQuantizationParameters()):
        super().__init__()
        self.algorithms_to_created = quantization_parameters.algorithms

    def _apply(self, model: ModelType, engine: Engine, statistic_points: StatisticPointsContainer) -> ModelType:
        for algorithm in self.algorithms:
            model = algorithm.apply(model, engine, statistic_points)
        return model

    def get_statistic_points(self, model: ModelType) -> StatisticPointsContainer:
        output = StatisticPointsContainer()
        for algorithm in self.algorithms:
            for statistic_points in algorithm.get_statistic_points(model).values():
                for statistic_point in statistic_points:
                    output.add_statistic_point(statistic_point)
        return output

    def create_subalgorithms(self) -> None:
        for algorithm, parameters in self.algorithms_to_created.items():
            if algorithm == PostTrainingAlgorithms.MinMaxQuantization:
                min_max_algo = MinMaxQuantization(parameters)
                self.algorithms.append(min_max_algo)
            if algorithm == PostTrainingAlgorithms.FastBiasCorrection:
                fast_bc_algo = FastBiasCorrection(parameters)
                self.algorithms.append(fast_bc_algo)
