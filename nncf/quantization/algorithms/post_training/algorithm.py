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

from typing import Dict, List, Optional, TypeVar, Union

from copy import deepcopy

from nncf import Dataset
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.logging import nncf_logger
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.algorithm import AlgorithmParameters
from nncf.quantization.algorithms.definitions import Granularity
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrectionParameters
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrectionParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantizationParameters
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

TModel = TypeVar('TModel')


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
                 ignored_scopes: Optional[List[str]] = None,
                 fast_bias_correction: bool = True,
                 ):
        self.algorithms = {MinMaxQuantization: MinMaxQuantizationParameters(
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
        )}

        bias_correction_algo = {BiasCorrection: BiasCorrectionParameters(
            number_samples=number_samples
        )}

        if fast_bias_correction:
            bias_correction_algo = {FastBiasCorrection: FastBiasCorrectionParameters(
                number_samples=number_samples
            )}
        self.algorithms.update(bias_correction_algo)

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        pass


class PostTrainingQuantization(Algorithm):
    """
    Implements Post-Training Quantization algorithm, which basically includes:
    1) MinMaxQuantization
    2) FastBiasCorrection or BiasCorrection
    3) ChannelAlignment

    Disclaimer: currently, it only supports MinMaxQuantization, FastBiasCorrection & BiasCorrection.
    ChannelAlignment will be added soon.

    """

    def __init__(self,
                 quantization_parameters: PostTrainingQuantizationParameters = PostTrainingQuantizationParameters()):
        super().__init__()
        self.algorithms = self._get_sub_algorithms(quantization_parameters.algorithms)

    @staticmethod
    def _get_sub_algorithms(algorithms: Dict[Algorithm, AlgorithmParameters]) -> List[Algorithm]:
        """
        This methods initializes sub-algorithms based on the general parameters.

        :param algorithms: The dictonary of the parameters per algorithm type.

        :return: The list of the algorithms instances.
        """
        algorithms_list = []
        for algorithm, parameters in algorithms.items():
            if algorithm == MinMaxQuantization:
                min_max_algo = MinMaxQuantization(parameters)
                algorithms_list.append(min_max_algo)
            if algorithm == FastBiasCorrection:
                fast_bc_algo = FastBiasCorrection(parameters)
                algorithms_list.append(fast_bc_algo)
            if algorithm == BiasCorrection:
                bc_algo = BiasCorrection(parameters)
                algorithms_list.append(bc_algo)
        return algorithms_list

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        output = StatisticPointsContainer()
        for algorithm in self.algorithms:
            for statistic_points in algorithm.get_statistic_points(model).values():
                for statistic_point in statistic_points:
                    output.add_statistic_point(statistic_point)
        return output

    def _create_statistics_aggregator(self,
                                      dataset: Dataset,
                                      backend: BackendType) -> StatisticsAggregator:
        """
        Creates backend-specific StatisticsAggregator.

        :param engine: engine for the model execution
        :param dataset: dataset for the statistics collection and validation
        :param model_transformer: backend-specific ModelTransformerBase instance
        :param backend: model backend type for the further differentiations
        :return: backnd-specific StatisticsAggregator
        """
        if backend == BackendType.ONNX:
            from nncf.experimental.onnx.statistics.aggregator import \
                ONNXStatisticsAggregator
            return ONNXStatisticsAggregator(dataset)
        return None

    def _apply(self,
               model: TModel,
               statistic_points: Optional[StatisticPointsContainer] = None,
               dataset: Optional[Dataset] = None) -> TModel:

        modified_model = deepcopy(model)
        if statistic_points is None:
            backend = get_backend(modified_model)

            # TODO (KodiaqQ): Remove after ONNX is removed from experimental
            if backend == BackendType.ONNX:
                nncf_logger.warning('You are using the experimental ONNX backend for post-training quantization.')

            statistics_aggregator = self._create_statistics_aggregator(dataset, backend)
            for algorithm in self.algorithms:
                algo_statistic_points = algorithm.get_statistic_points(modified_model)
                statistics_aggregator.register_stastistic_points(algo_statistic_points)

            statistics_aggregator.collect_statistics(modified_model)
            statistic_points = statistics_aggregator.statistic_points

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(modified_model, statistic_points)
        return modified_model
