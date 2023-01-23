"""
 Copyright (c) 2023 Intel Corporation
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

import numpy as np

from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import BatchTensorStatistic


class ONNXMinMaxTensorStatistic(MinMaxTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: np.ndarray, tensor2: np.ndarray, rtol=1e-6) -> bool:
        return bool(np.allclose(tensor1, tensor2, rtol=rtol))


class ONNXMeanTensorStatistic(MeanTensorStatistic):
    @staticmethod
    def tensor_eq(tensor: np.ndarray, rtol=1e-6) -> bool:
        return bool(np.all(tensor, rtol=rtol))


class ONNXBatchTensorStatistic(BatchTensorStatistic):
    @staticmethod
    def tensor_eq(tensor: np.ndarray, rtol=1e-6) -> bool:
        return bool(np.all(tensor, rtol=rtol))
