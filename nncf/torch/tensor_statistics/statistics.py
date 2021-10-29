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

from abc import abstractmethod
from collections import Counter

import torch

from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic


class PTTensorStatistic(TensorStatistic):
    @staticmethod
    def torch_tensor_eq(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-6) -> bool:
        return bool(torch.allclose(tensor1, tensor2, rtol=rtol))

    @abstractmethod
    def __eq__(self, other):
        pass


class PTMinMaxTensorStatistic(MinMaxTensorStatistic, PTTensorStatistic):
    def __eq__(self, other: 'PTMinMaxTensorStatistic') -> bool:
        return self.torch_tensor_eq(self.min_values, other.min_values) and \
               self.torch_tensor_eq(self.max_values, other.max_values)


class PTMedianMADTensorStatistic(MedianMADTensorStatistic, PTTensorStatistic):
    def __eq__(self, other: 'PTMedianMADTensorStatistic') -> bool:
        return self.torch_tensor_eq(self.median_values, other.median_values) and \
               self.torch_tensor_eq(self.mad_values, other.mad_values)


class PTPercentileTensorStatistic(PercentileTensorStatistic, PTTensorStatistic):
    def __eq__(self, other: 'PTPercentileTensorStatistic', rtol=1e-9) -> bool:
        if Counter(self.percentile_vs_values_dict.keys()) != Counter(other.percentile_vs_values_dict.keys()):
            return False
        for pct in self.percentile_vs_values_dict.keys():
            if not self.torch_tensor_eq(self.percentile_vs_values_dict[pct],
                                        other.percentile_vs_values_dict[pct]):
                return False
        return True
