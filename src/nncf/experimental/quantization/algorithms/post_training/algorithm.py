# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Callable, Optional, TypeVar

from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.experimental.quantization.algorithms.post_training.pipeline import experimental_create_ptq_pipeline
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.quantization.advanced_parameters import AdvancedBiasCorrectionParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.advanced_parameters import RangeEstimatorParameters
from nncf.quantization.algorithms.algorithm import Algorithm

TModel = TypeVar("TModel")
TPass = Callable[[TModel], TModel]


class ExperimentalPostTrainingQuantization(Algorithm):
    """
    Implements Experimental Post-Training Quantization algorithm, which basically includes:
    1) ChannelAlignment
    2) MinMaxRangeInit
    3) FastBiasCorrection or BiasCorrection
    """

    def __init__(
        self,
        quantizer: Quantizer,
        subset_size: int = 300,
        fast_bias_correction: Optional[bool] = True,
        smooth_quant: bool = False,
        bias_correction_params: Optional[AdvancedBiasCorrectionParameters] = None,
        smooth_quant_params: Optional[AdvancedSmoothQuantParameters] = None,
        activations_range_estimator_params: Optional[RangeEstimatorParameters] = None,
        weights_range_estimator_params: Optional[RangeEstimatorParameters] = None,
        batchwise_statistics: bool = False,
    ):
        """
        :param quantizer: Quantizer to use in MiMaxRangeInit algorithm.
        :param subset_size: Size of a subset to calculate activations
            statistics used for quantization.
        :param fast_bias_correction: Setting this option to `False` enables a different
            bias correction method which is more accurate, in general, and takes
            more time but requires less memory. None disables the bias correction algorithm.
        :param smooth_quant: Setting this option to `True` enables the SmoothQuant algorithm.
        :param bias_correction_params: Contains advanced parameters for fine-tuning bias correction algorithm.
        :param smooth_quant_params: Contains advanced alpha parameters for SmoothQuant algorithm.
        :param activations_range_estimator_params: Contains parameters for estimating the range
            of activations of the model.
        :param weights_range_estimator_params: Contains parameters for estimating the range
            of weights of the model.
        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch, default is False.
        """
        self._pipeline = experimental_create_ptq_pipeline(
            quantizer=quantizer,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            smooth_quant=smooth_quant,
            bias_correction_params=bias_correction_params,
            smooth_quant_params=smooth_quant_params,
            activations_range_estimator_params=activations_range_estimator_params,
            weights_range_estimator_params=weights_range_estimator_params,
            batchwise_statistics=batchwise_statistics,
        )

    @property
    def available_backends(self) -> list[BackendType]:
        backends = set(BackendType)
        for algorithm in itertools.chain.from_iterable(self._pipeline.pipeline_steps):
            backends = backends.intersection(algorithm.available_backends)
        return list(backends)

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        return self._pipeline.get_statistic_points_for_step(0, model, graph)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        if dataset is None and len(self._pipeline.pipeline_steps) > 1:
            msg = (
                "A dataset is required for the post-training quantization "
                "algorithm to collect statistics for intermediate models."
            )
            raise ValueError(msg)

        step_index_to_statistics = None
        if statistic_points:
            step_index_to_statistics = {0: statistic_points}

        return self._pipeline.run_from_step(model, dataset, graph, 0, step_index_to_statistics)
