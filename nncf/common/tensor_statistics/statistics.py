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

from abc import ABC
from abc import abstractmethod
from collections import Counter
from typing import TypeVar

TensorType = TypeVar("TensorType")


class TensorStatistic(ABC):
    """Base class that stores statistic data"""

    @staticmethod
    @abstractmethod
    def tensor_eq(tensor1: TensorType, tensor2: TensorType, rtol=1e-6) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class MinMaxTensorStatistic(TensorStatistic):
    MIN_STAT = "min_values"
    MAX_STAT = "max_values"

    def __init__(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values

    def __eq__(self, other: "MinMaxTensorStatistic") -> bool:
        return self.tensor_eq(self.min_values, other.min_values) and self.tensor_eq(self.max_values, other.max_values)


class MeanTensorStatistic(TensorStatistic):
    MEAN_STAT = "mean_values"
    SHAPE_STAT = "shape"

    """
    Base class for the statistics that collects as mean per-axis
    """

    def __init__(self, mean_values, shape):
        """
        :param mean_values: Сollected mean per-axis values.
        :param shape: The shape of the collected statistics.
        """
        self.mean_values = mean_values
        self.shape = shape

    def __eq__(self, other: "MeanTensorStatistic") -> bool:
        return self.tensor_eq(self.mean_values, other.mean_values) and self.tensor_eq(self.shape, other.shape)


class MedianMADTensorStatistic(TensorStatistic):
    def __init__(self, median_values, mad_values):
        self.median_values = median_values
        self.mad_values = mad_values

    def __eq__(self, other: "MedianMADTensorStatistic") -> bool:
        return self.tensor_eq(self.median_values, other.median_values) and self.tensor_eq(
            self.mad_values, other.mad_values
        )


class PercentileTensorStatistic(TensorStatistic):
    def __init__(self, percentile_vs_values_dict):
        self.percentile_vs_values_dict = percentile_vs_values_dict

    def __eq__(self, other: "PercentileTensorStatistic", rtol=1e-9) -> bool:
        if Counter(self.percentile_vs_values_dict.keys()) != Counter(other.percentile_vs_values_dict.keys()):
            return False
        for pct in self.percentile_vs_values_dict.keys():
            if not self.tensor_eq(self.percentile_vs_values_dict[pct], other.percentile_vs_values_dict[pct]):
                return False
        return True


class BatchTensorStatistic(TensorStatistic):
    VALUES_STATS = "values"

    """
    Base class for the statistics that collects as mean per-batch
    """

    def __init__(self, values):
        """
        :param values: Сollected per-batch values.
        """
        self.values = values

    def __eq__(self, other: "BatchTensorStatistic") -> bool:
        return self.tensor_eq(self.values, other.values)
