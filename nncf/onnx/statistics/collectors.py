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

from typing import Any, Callable, Deque, List, Optional, Union

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import BatchStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.onnx.statistics.statistics import ONNXBatchTensorStatistic
from nncf.onnx.statistics.statistics import ONNXMeanTensorStatistic
from nncf.onnx.statistics.statistics import ONNXMinMaxTensorStatistic
from nncf.onnx.tensor import ONNXNNCFTensor


class ONNXNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for ONNXNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple, list], keepdims: bool = False) -> NNCFTensor:
        return ONNXNNCFTensor(np.amin(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple, list], keepdims: bool = False) -> NNCFTensor:
        return ONNXNNCFTensor(np.amax(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.abs(x.tensor))

    @staticmethod
    def min(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.minimum(x1.tensor, x2.tensor))

    @staticmethod
    def max(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.maximum(x1.tensor, x2.tensor))

    @staticmethod
    def mean(x: NNCFTensor, axis: Union[int, tuple, list], keepdims=False) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, tuple, list], keepdims=False) -> NNCFTensor:
        return ONNXNNCFTensor(np.median(x.tensor, axis=axis, keepdims=keepdims))

    @classmethod
    def masked_mean(
        cls, x: NNCFTensor, axis: Optional[Union[int, tuple, list]], mask: Optional[NNCFTensor], keepdims: bool = False
    ) -> NNCFTensor:
        if mask is None:
            return cls.mean(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return ONNXNNCFTensor(np.ma.mean(masked_x, axis=axis, keepdims=False).data)

    @classmethod
    def masked_median(
        cls, x: NNCFTensor, axis: Optional[Union[int, tuple, list]], mask: Optional[NNCFTensor], keepdims: bool = False
    ) -> NNCFTensor:
        if mask is None:
            return cls.median(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return ONNXNNCFTensor(np.ma.median(masked_x, axis=axis, keepdims=keepdims).data)

    @classmethod
    def no_outliers_map(
        cls,
        x: NNCFTensor,
        fn: Callable[[NNCFTensor, int, NNCFTensor], Any],
        axis: int = 0,
        alpha: float = 0.01,
        keepdims: bool = False,
    ) -> NNCFTensor:
        if len(x.shape) == 1:
            return fn(x, axis=None, mask=None, keepdims=keepdims)

        x = x.tensor
        if axis:
            x = np.moveaxis(x, axis, 0)

        low_values, high_values = np.quantile(x, [alpha, 1 - alpha], 0)
        outliers_mask = np.logical_or(x < low_values, high_values < x)
        return fn(ONNXNNCFTensor(x), axis=0, mask=ONNXNNCFTensor(outliers_mask), keepdims=keepdims)

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return ONNXNNCFTensor(np.stack(x, axis=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        return [ONNXNNCFTensor(np.squeeze(e, axis)) for e in np.split(x.tensor, x.tensor.shape[axis], axis=axis)]

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return np.sum(tensor.tensor)

    @staticmethod
    def quantile(
        tensor: NNCFTensor, quantile: Union[float, List[float]], axis: Union[int, tuple, list], keepdims: bool = False
    ) -> List[TensorElementsType]:
        result = np.quantile(tensor.tensor, quantile, axis, keepdims=keepdims)
        return [ONNXNNCFTensor(x) for x in result]

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return ONNXNNCFTensor(np.mean(x.tensor, axis=0))
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return ONNXNNCFTensor(np.mean(t, axis=(0, 2)))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=0, keepdims=True))


class ONNXMinMaxStatisticCollector(MinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMinMaxTensorStatistic:
        return ONNXMinMaxTensorStatistic(self._min_values.tensor, self._max_values.tensor)


class ONNXMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMinMaxTensorStatistic:
        return ONNXMinMaxTensorStatistic(self._min_aggregate().tensor, self._max_aggregate().tensor)


class ONNXMeanStatisticCollector(MeanStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMeanTensorStatistic:
        return ONNXMeanTensorStatistic(self._mean_aggregate().tensor, self._shape())


class ONNXBatchStatisticCollector(BatchStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXBatchTensorStatistic:
        return ONNXBatchTensorStatistic(self._all_values)
