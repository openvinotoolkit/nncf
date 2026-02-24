# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, TypeVar

from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.range_estimator import RangeEstimatorParameters

TModel = TypeVar("TModel")


class MinMaxRangeEstimator(Algorithm):
    def __init__(
        self,
        quantizer: Quantizer,
        subset_size: int = 300,
        inplace_statistics: bool = True,
        batchwise_statistics: bool = False,
        activations_range_estimator_params: Optional[RangeEstimatorParameters] = None,
        weights_range_estimator_params: Optional[RangeEstimatorParameters] = None,
    ):
        """
        :param quantizer: Instance of Quantizer to retrieve a quantization config
            for the given model.
        :param subset_size: Size of a subset to calculate activations statistics used
            for quantization, defaults to 300.
        :param inplace_statistics: Defines whether to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch, default is False.
        :param activations_range_estimator_params: Quantization range estimation
            parameters for activation.
        :param weights_range_estimator_params: Quantization range estimation parameters
            for weights.
        """
        self._quantizer = quantizer
        self._min_max_algo = MinMaxQuantization(
            subset_size=subset_size,
            inplace_statistics=inplace_statistics,
            batchwise_statistics=batchwise_statistics,
            activations_range_estimator_params=activations_range_estimator_params,
            weights_range_estimator_params=weights_range_estimator_params,
        )

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.TORCH_FX]

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        if self._min_max_algo._quantization_target_points_to_qconfig is None:
            msg = (
                "Statistic points are not available."
                " Please call `get_statistic_points` before calling the `apply` method."
            )
            raise RuntimeError(msg)
        return self._min_max_algo.apply(model=model, graph=graph, statistic_points=statistic_points)

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        quantizer_setup = self._quantizer.get_quantization_setup(model, graph)
        self._min_max_algo._set_backend_entity(model)
        self._min_max_algo._init_cache()
        self._min_max_algo.fill_quantization_target_points(quantizer_setup, graph)
        return self._min_max_algo.get_cached_statistic_points(model, graph)
