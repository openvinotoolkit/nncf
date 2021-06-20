"""
 Copyright (c) 2021 Intel Corporation
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

from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict

import torch


class TensorStatistic(ABC):
    @staticmethod
    def torch_tensor_eq(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-6) -> bool:
        return bool(torch.allclose(tensor1, tensor2, rtol=rtol))

    @abstractmethod
    def __eq__(self, other):
        pass


class MinMaxTensorStatistic(TensorStatistic):
    def __init__(self, min_values: torch.Tensor, max_values: torch.Tensor):
        self.min_values = min_values
        self.max_values = max_values

    @classmethod
    def from_stat(cls, statistic: TensorStatistic):
        if isinstance(statistic, MinMaxTensorStatistic):
            return cls(statistic.min_values, statistic.max_values)
        if isinstance(statistic, MedianMADTensorStatistic):
            # Using three-sigma approach
            # Constant factor depends on the distribution form - assuming normal and the factor 1.4826
            return cls(statistic.median_values - 3 * 1.4826 * statistic.mad_values,
                       statistic.median_values + 3 * 1.4826 * statistic.mad_values)
        if isinstance(statistic, PercentileTensorStatistic):
            if len(statistic.percentile_vs_values_dict.keys()) < 2:
                raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
            min_pct = min(statistic.percentile_vs_values_dict.keys())
            max_pct = max(statistic.percentile_vs_values_dict.keys())
            return cls(statistic.percentile_vs_values_dict[min_pct],
                       statistic.percentile_vs_values_dict[max_pct])
        raise ValueError("Unknown statistic to generate min-max stat from!")

    def __eq__(self, other: 'MinMaxTensorStatistic') -> bool:
        return self.torch_tensor_eq(self.min_values, other.min_values) and \
               self.torch_tensor_eq(self.max_values, other.max_values)


class MedianMADTensorStatistic(TensorStatistic):
    def __init__(self, median_values: torch.Tensor, mad_values: torch.Tensor):
        self.median_values = median_values
        self.mad_values = mad_values

    def __eq__(self, other: 'MedianMADTensorStatistic') -> bool:
        return self.torch_tensor_eq(self.median_values, other.median_values) and \
               self.torch_tensor_eq(self.mad_values, other.mad_values)


class PercentileTensorStatistic(TensorStatistic):
    def __init__(self, percentile_vs_values_dict: Dict[float, torch.Tensor]):
        self.percentile_vs_values_dict = percentile_vs_values_dict

    def __eq__(self, other: 'PercentileTensorStatistic', rtol=1e-9) -> bool:
        if Counter(self.percentile_vs_values_dict.keys()) != Counter(other.percentile_vs_values_dict.keys()):
            return False
        for pct in self.percentile_vs_values_dict.keys():
            if not self.torch_tensor_eq(self.percentile_vs_values_dict[pct],
                                        other.percentile_vs_values_dict[pct]):
                return False
        return True
