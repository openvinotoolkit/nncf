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
from __future__ import annotations

from typing import Union

from nncf.common.graph.utils import get_reduction_axes
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.tensor_statistics.collectors import AggregationAxes
from nncf.common.tensor_statistics.collectors import ReductionAxes


class RangeInitCollectorParams:
    """
    Defines low-level parameters that are used to instantiate statistic collectors.
    """

    def __init__(self, is_weights: bool, scheme: QuantizationScheme, per_channel: bool):
        """
        Initializes Range Initialization Collector Parameters.

        :param is_weights: Boolean that defines tensor type. True for Weights, False for Activations.
        :param scheme: Quantization scheme: symmetric or asymmetric.
        :param per_channel: Quantization granularity.
        """
        self._is_weights = is_weights
        self._scheme = scheme
        self._is_per_channel = per_channel

    @property
    def is_weights(self) -> bool:
        """
        Returns boolean that defines tensor type.
        True for Weights, False for Activations.
        """
        return self._is_weights

    @property
    def scheme(self) -> QuantizationScheme:
        """
        Returns quantization scheme: symmetric or asymmetric.
        """
        return self._scheme

    @property
    def is_per_channel(self) -> bool:
        """
        Returns quantization granularity.
        """
        return self._is_per_channel

    def use_per_sample_stats(self, per_sample_stats: bool) -> bool:
        """
        For activations, if per_sample_stats is True, statistics will be collected per-sample.
        For weights statistics are always collected per-batch.

        :param per_sample_stats: Defined by certain collector design.
        :return: A boolean that defines whether to collect statistics per-sample or per-batch.
        """
        return per_sample_stats and (not self._is_weights)

    @property
    def use_abs_max(self) -> bool:
        """Applies abs(max) for symmetric quantization."""
        return self._scheme == QuantizationScheme.SYMMETRIC

    @property
    def use_means_of_mins(self) -> bool:
        return not self._is_weights and not self._is_per_channel and self._scheme == "asymmetric"

    @property
    def use_means_of_maxs(self) -> bool:
        return not self._is_weights and not self._is_per_channel

    def _get_reduction_axes(
        self,
        shape_to_reduce: Union[tuple[int, ...], list[int]],
        quantization_axes: Union[tuple[int, ...], list[int]],
        aggregation_axes: Union[tuple[int, ...], list[int]],
    ) -> tuple[int, ...]:
        """
        Returns axes for a reducer regarding aggregation axes. As aggregator takes axes counting from stacked tensors,
        from these axes only tensor related axes should be used for reducer.

        :param shape_to_reduce: Shape of a reduced tensor.
        :param quantization_axes: Axes of quantization.
        :param aggregation_axes: Axes of aggregator which is applied onto reduced tensor.
        :return: Axes for reducer.
        """
        axes_to_keep = set(el - 1 for el in aggregation_axes if el != 0)
        axes_to_keep.update(quantization_axes)
        return get_reduction_axes(list(axes_to_keep), shape_to_reduce)

    def _get_aggregation_axes(self, batchwise_statistics: bool) -> tuple[int, ...]:
        """
        Returns axes for aggregator.

        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch.
        :return Tuple[int]: Aggregation axes.
        """
        return (0, 1) if batchwise_statistics else (0,)

    def get_reduction_aggregation_axes(
        self,
        shape_to_reduce: Union[tuple[int, ...], list[int]],
        quantization_axes: Union[tuple[int, ...], list[int]],
        batchwise_statistics: bool,
    ) -> tuple[ReductionAxes, AggregationAxes]:
        """
        Calculates the reduction axes, aggregation axes for the tensor.

        :param shape_to_reduce: Shape of the tensor.
        :param quantization_axes: Quantization axes if per-channel quantization.
        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch.
        :return: Reduction axes and aggregation axes.
        """
        aggregation_axes = self._get_aggregation_axes(batchwise_statistics)
        reduction_axes = self._get_reduction_axes(shape_to_reduce, quantization_axes, aggregation_axes)
        return reduction_axes, aggregation_axes
