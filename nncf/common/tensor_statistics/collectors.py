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

from abc import ABC
from abc import abstractmethod
from collections import deque
from typing import Tuple, Set


ReductionShape = Tuple[int]


class TensorStatisticCollectorBase(ABC):
    def __init__(self, reduction_shape: ReductionShape = None, num_samples: int = None):
        self._reduction_shape = reduction_shape
        self._enabled = True
        self._collected_samples = 0
        self._num_samples = num_samples

    def register_input(self, x):
        if not self._enabled or \
                self._num_samples is not None and self._collected_samples >= self._num_samples:
            return x
        if self._reduction_shape is None:
            self._reduction_shape = {tuple(x.shape)}
        self._register_input(x)
        self._collected_samples += 1
        return x

    @abstractmethod
    def _register_input(self, x):
        pass

    def get_statistics(self):
        if self._collected_samples == 0:
            raise StatisticsNotCollectedError()
        return self._get_statistics()

    @abstractmethod
    def _get_statistics(self):
        pass

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset(self):
        self._collected_samples = 0
        self._reset()

    @abstractmethod
    def _reset(self):
        pass

    def collected_samples(self) -> int:
        return self._collected_samples


class StatisticsNotCollectedError(Exception):
    pass


class OnlineTensorStatisticCollector(TensorStatisticCollectorBase, ABC):
    pass


class OfflineTensorStatisticCollector(TensorStatisticCollectorBase, ABC):
    def __init__(self, reduction_shape: ReductionShape = None, num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples)
        self._samples = deque(maxlen=window_size)

    def _reset(self):
        self._samples.clear()
