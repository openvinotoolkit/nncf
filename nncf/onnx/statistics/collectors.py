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

from typing import Deque, List, Optional, Tuple, Union

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import RawStatisticCollector
from nncf.onnx.statistics.statistics import ONNXMeanTensorStatistic
from nncf.onnx.statistics.statistics import ONNXMinMaxTensorStatistic
from nncf.onnx.statistics.statistics import ONNXRawTensorStatistic
from nncf.onnx.tensor import ONNXNNCFTensor


class ONNXNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for ONNXNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        return ONNXNNCFTensor(np.amin(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
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
    def mean(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> NNCFTensor:
        return ONNXNNCFTensor(np.median(x.tensor, axis=axis, keepdims=keepdims))

    @classmethod
    def masked_mean(
        cls,
        x: NNCFTensor,
        axis: Optional[Union[int, Tuple[int, ...], List[int]]],
        mask: Optional[NNCFTensor],
        keepdims: bool = False,
    ) -> NNCFTensor:
        if mask is None:
            return cls.mean(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return ONNXNNCFTensor(np.ma.mean(masked_x, axis=axis, keepdims=False).data)

    @classmethod
    def masked_median(
        cls,
        x: NNCFTensor,
        axis: Optional[Union[int, Tuple[int, ...], List[int]]],
        mask: Optional[NNCFTensor],
        keepdims: bool = False,
    ) -> NNCFTensor:
        if mask is None:
            return cls.median(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return ONNXNNCFTensor(np.ma.median(masked_x, axis=axis, keepdims=keepdims).data)

    @staticmethod
    def logical_or(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.logical_or(input_.tensor, other.tensor))

    @staticmethod
    def less(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(input_.tensor < other.tensor)

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return ONNXNNCFTensor(np.stack(x, axis=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        return [ONNXNNCFTensor(np.squeeze(e, axis)) for e in np.split(x.tensor, x.tensor.shape[axis], axis=axis)]

    @staticmethod
    def squeeze(x: NNCFTensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> NNCFTensor:
        raise NotImplementedError()

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return np.sum(tensor.tensor)

    @staticmethod
    def quantile(
        tensor: NNCFTensor,
        quantile: Union[float, List[float]],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        result = np.quantile(tensor.tensor, quantile, axis, keepdims=keepdims)
        return [ONNXNNCFTensor(x) for x in result]

    @classmethod
    def percentile(
        cls,
        tensor: NNCFTensor,
        percentile: Union[float, List[float]],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        raise NotImplementedError()

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return ONNXNNCFTensor(np.mean(x.tensor, axis=0))
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return ONNXNNCFTensor(np.mean(t, axis=(0, 2)))

    @staticmethod
    def transpose(x: NNCFTensor, axes: Tuple[int, ...]) -> NNCFTensor:
        return ONNXNNCFTensor(np.transpose(x.tensor, axes))

    @staticmethod
    def reshape(x: NNCFTensor, shape: Tuple[int, ...]) -> NNCFTensor:
        return ONNXNNCFTensor(np.reshape(x.tensor, shape))

    @staticmethod
    def cat(x: List[NNCFTensor], axis: int) -> NNCFTensor:
        x = [t.tensor for t in x]
        return ONNXNNCFTensor(np.concatenate(x, axis))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def sub(a: NNCFTensor, b: NNCFTensor) -> NNCFTensor:
        raise NotImplementedError()

    @staticmethod
    def zero_elements(x: NNCFTensor) -> NNCFTensor:
        raise NotImplementedError()


class ONNXMinMaxStatisticCollector(MinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMinMaxTensorStatistic:
        return ONNXMinMaxTensorStatistic(
            min_values=self._min_values.tensor,
            max_values=self._max_values.tensor,
        )


class ONNXMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMinMaxTensorStatistic:
        return ONNXMinMaxTensorStatistic(
            min_values=self._min_aggregate().tensor,
            max_values=self._max_aggregate().tensor,
        )


class ONNXMeanStatisticCollector(MeanStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMeanTensorStatistic:
        return ONNXMeanTensorStatistic(
            mean_values=self._mean_aggregate().tensor,
            shape=self._shape(),
        )


class ONNXRawStatisticCollector(RawStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXRawTensorStatistic:
        return ONNXRawTensorStatistic(self._all_values)
