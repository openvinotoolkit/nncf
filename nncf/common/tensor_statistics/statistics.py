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

from abc import ABC
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, TypeVar, cast

from nncf.tensor import Tensor
from nncf.tensor import functions as fns

TensorType = TypeVar("TensorType")


class TensorStatistic(ABC):
    """Base class that stores statistic data"""

    TENSOR_STATISTIC_OUTPUT_KEY = "tensor_statistic_output"

    @staticmethod
    def tensor_eq(tensor1: Tensor, tensor2: Tensor, rtol: float = 1e-6) -> bool:
        return cast(bool, fns.allclose(tensor1, tensor2))

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class MinMaxTensorStatistic(TensorStatistic):
    MIN_STAT = "min_values"
    MAX_STAT = "max_values"

    def __init__(self, min_values: Tensor, max_values: Tensor):
        self.min_values = min_values
        self.max_values = max_values

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MinMaxTensorStatistic):
            return False
        return self.tensor_eq(self.min_values, other.min_values) and self.tensor_eq(self.max_values, other.max_values)


class MeanTensorStatistic(TensorStatistic):
    MEAN_STAT = "mean_values"
    SHAPE_STAT = "shape"

    """
    Base class for the statistics that collects as mean per-axis
    """

    def __init__(self, mean_values: Tensor, shape: Tensor):
        """
        :param mean_values: Collected mean per-axis values.
        :param shape: The shape of the collected statistics.
        """
        self.mean_values = mean_values
        self.shape = shape

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MeanTensorStatistic):
            return False
        return self.tensor_eq(self.mean_values, other.mean_values) and self.tensor_eq(self.shape, other.shape)


class MedianMADTensorStatistic(TensorStatistic):
    MEDIAN_VALUES_STAT = "median_values"
    MAD_VALUES_STAT = "mad_values"

    def __init__(self, median_values: Tensor, mad_values: Tensor):
        self.median_values = median_values
        self.mad_values = mad_values

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MedianMADTensorStatistic):
            return False
        return self.tensor_eq(self.median_values, other.median_values) and self.tensor_eq(
            self.mad_values, other.mad_values
        )


class PercentileTensorStatistic(TensorStatistic):
    PERCENTILE_VS_VALUE_DICT = "percentile_vs_values_dict"

    def __init__(self, percentile_vs_values_dict: Dict[float, Any]):
        self.percentile_vs_values_dict = percentile_vs_values_dict

    def __eq__(self, other: Any, rtol: float = 1e-9) -> bool:
        if not isinstance(other, PercentileTensorStatistic):
            return False
        if Counter(self.percentile_vs_values_dict.keys()) != Counter(other.percentile_vs_values_dict.keys()):
            return False
        for pct in self.percentile_vs_values_dict:
            if not self.tensor_eq(self.percentile_vs_values_dict[pct], other.percentile_vs_values_dict[pct]):
                return False
        return True


class RawTensorStatistic(TensorStatistic):
    VALUES_STATS = "values"

    """
    Base class for the raw statistics, without any aggregation.
    """

    def __init__(self, values: Tensor):
        """
        :param values: Collected raw values.
        """
        self.values = values

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RawTensorStatistic):
            return False
        return self.tensor_eq(self.values, other.values)
