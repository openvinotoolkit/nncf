# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, TypeVar

import numpy as np

from nncf import Dataset
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.bias_correction.algorithm import BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FAST_BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")


class PostTrainingQuantization(Algorithm):
    """
    Implements Post-Training Quantization algorithm, which basically includes:
    1) ChannelAlignment
    2) MinMaxQuantization
    3) FastBiasCorrection or BiasCorrection

    Disclaimer: currently, it only supports MinMaxQuantization, FastBiasCorrection & BiasCorrection.
    ChannelAlignment will be added soon.
    """

    def __init__(
        self,
        preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
        target_device: TargetDevice = TargetDevice.ANY,
        subset_size: int = 300,
        fast_bias_correction: bool = True,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
    ):
        """
        :param preset: A preset that controls the quantization mode
            (symmetric and asymmetric). It can take the following values:
            - `performance`: Symmetric quantization of weights and activations.
            - `mixed`: Symmetric quantization of weights and asymmetric
            quantization of activations.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device.
        :param subset_size: Size of a subset to calculate activations
            statistics used for quantization.
        :param fast_bias_correction: Setting this option to `False` enables a different
            bias correction method which is more accurate, in general, and takes
            more time but requires less memory.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param advanced_parameters: Advanced quantization parameters for
            fine-tuning the quantization algorithm
        """
        super().__init__()
        self.algorithms = []

        if advanced_parameters is None:
            advanced_parameters = AdvancedQuantizationParameters()

        min_max_quantization = MinMaxQuantization(
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            model_type=model_type,
            ignored_scope=ignored_scope,
            overflow_fix=advanced_parameters.overflow_fix,
            quantize_outputs=advanced_parameters.quantize_outputs,
            inplace_statistics=advanced_parameters.inplace_statistics,
            activations_quantization_params=advanced_parameters.activations_quantization_params,
            weights_quantization_params=advanced_parameters.weights_quantization_params,
            activations_range_estimator_params=advanced_parameters.activations_range_estimator_params,
            weights_range_estimator_params=advanced_parameters.weights_range_estimator_params,
            backend_params=advanced_parameters.backend_params,
        )

        self.algorithms.append(min_max_quantization)

        if advanced_parameters.disable_bias_correction:
            return

        bias_correction_params = advanced_parameters.bias_correction_params
        if fast_bias_correction:
            threshold = FAST_BIAS_CORRECTION_THRESHOLD
            if bias_correction_params.threshold is not None:
                threshold = bias_correction_params.threshold
            bias_correction = FastBiasCorrection(
                subset_size=subset_size,
                threshold=threshold,
                apply_for_all_nodes=bias_correction_params.apply_for_all_nodes,
                inplace_statistics=advanced_parameters.inplace_statistics,
                backend_params=advanced_parameters.backend_params,
            )
        else:
            threshold = BIAS_CORRECTION_THRESHOLD
            if bias_correction_params.threshold is not None:
                threshold = bias_correction_params.threshold
            bias_correction_subset_size = max(np.int(subset_size * 0.2), 1)
            bias_correction = BiasCorrection(
                subset_size=bias_correction_subset_size,
                threshold=threshold,
                apply_for_all_nodes=bias_correction_params.apply_for_all_nodes,
                inplace_statistics=advanced_parameters.inplace_statistics,
                backend_params=advanced_parameters.backend_params,
            )

        self.algorithms.append(bias_correction)

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

    def _create_statistics_aggregator(self, dataset: Dataset, backend: BackendType) -> StatisticsAggregator:
        """
        Creates backend-specific StatisticsAggregator.

        :param engine: Engine for the model execution
        :param dataset: Dataset for the statistics collection and validation
        :param model_transformer: Backend-specific ModelTransformerBase instance
        :param backend: Model backend type for the further differentiations
        :return: Backend-specific StatisticsAggregator
        """
        if backend == BackendType.ONNX:
            from nncf.onnx.statistics.aggregator import ONNXStatisticsAggregator

            return ONNXStatisticsAggregator(dataset)
        if backend == BackendType.OPENVINO:
            from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

            return OVStatisticsAggregator(dataset)
        if backend == BackendType.TORCH:
            from nncf.torch.statistics.aggregator import PTStatisticsAggregator

            return PTStatisticsAggregator(dataset)
        return None

    def _apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        modified_model = copy_model(model)
        if statistic_points is None:
            backend = get_backend(modified_model)

            statistics_aggregator = self._create_statistics_aggregator(dataset, backend)
            for algorithm in self.algorithms:
                algo_statistic_points = algorithm.get_statistic_points(modified_model)
                statistics_aggregator.register_statistic_points(algo_statistic_points)

            statistics_aggregator.collect_statistics(modified_model)
            statistic_points = statistics_aggregator.statistic_points

        for algorithm in self.algorithms:
            modified_model = algorithm.apply(modified_model, statistic_points)
        return modified_model
