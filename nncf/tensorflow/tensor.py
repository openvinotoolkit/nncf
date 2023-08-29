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
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import Union
from typing import Union

import numpy as np
import tensorflow as tf
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant

from nncf.common.tensor import DeviceType
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend
from nncf.common.tensor import TensorDtype
from nncf.common.tensor import WrappingIterator

_DTYPE_MAP: Dict[TensorDtype, tf.DType] = {TensorDtype.FLOAT32: tf.float32, TensorDtype.INT64: tf.int64}

_INV_DTYPE_MAP: Dict[tf.DType, TensorDtype] = {v: k for k, v in _DTYPE_MAP.items()}


class TFNNCFTensor(NNCFTensor[tf.Tensor]):
    # actually this builds upon EagerTensor, but the corresponding obj is marked __internal__
    @property
    def backend(self) -> Type["NNCFTensorBackend"]:
        return TFNNCFTensorBackend

    def _is_native_bool(self, bool_result: Any) -> bool:
        assert False  # TODO (vshampor): check whether this is relevant for TF

    @property
    def ndim(self) -> int:
        return len(self._tensor.shape)

    @property
    def shape(self) -> List[int]:
        return self._tensor.shape.as_list()

    @property
    def device(self) -> DeviceType:
        return self._tensor.device

    @property
    def size(self) -> int:
        return int(tf.size(self._tensor))

    def is_empty(self) -> bool:
        return self.size == 0

    def mean(self, axis: int, keepdims: bool = None) -> "TFNNCFTensor":
        return TFNNCFTensor(tf.math.reduce_mean(self._tensor, axis=axis, keepdims=keepdims))

    def median(self, axis: int = None, keepdims: bool = False) -> "TFNNCFTensor":
        # TODO (vshampor): looks like calculating median in TF is only possible through an extra dependency
        #  tensorflow_probability
        numpy_array: np.ndarray = self._tensor.numpy()
        return TFNNCFTensor(tf.convert_to_tensor(np.median(numpy_array, axis=axis, keepdims=keepdims)))

    def reshape(self, *shape: int) -> "TFNNCFTensor":
        return TFNNCFTensor(tf.reshape(self._tensor, shape))

    def to_numpy(self) -> np.ndarray:
        return self._tensor.numpy()

    def __iter__(self) -> Iterator:
        return WrappingIterator[TFNNCFTensor](iter(self._tensor))

    def matmul(self, other: "TFNNCFTensor") -> "TFNNCFTensor":
        return TFNNCFTensor(tf.matmul(self._tensor, other.tensor))

    def astype(self, dtype: TensorDtype) -> "TFNNCFTensor":
        return TFNNCFTensor(tf.cast(self._tensor, _INV_DTYPE_MAP[dtype]))

    @property
    def dtype(self) -> TensorDtype:
        return _DTYPE_MAP[self._tensor.dtype]

    def any(self) -> bool:
        return bool(tf.reduce_any(self._tensor))

    def all(self) -> bool:
        return bool(tf.reduce_all(self._tensor))

    def min(self) -> float:
        return tf.reduce_min(self._tensor).numpy().item()

    def max(self) -> float:
        return tf.reduce_max(self._tensor).numpy().item()


class TFNNCFTensorBackend(NNCFTensorBackend):
    inf = np.inf  # TF seems to reuse the np.inf value for representing own inf values

    @staticmethod
    def moveaxis(tensor: TFNNCFTensor, src: int, dst: int) -> TFNNCFTensor:
        return TFNNCFTensor(tf.experimental.numpy.moveaxis(tensor.tensor, src, dst))

    @staticmethod
    def mean(tensor: TFNNCFTensor, axis: Union[int, Tuple[int, ...]], keepdims: bool = False) -> TFNNCFTensor:
        return TFNNCFTensor(tf.reduce_mean(tensor.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def mean_of_list(tensor_list: List[TFNNCFTensor], axis: int) -> TFNNCFTensor:
        cated = tf.concat([t.tensor for t in tensor_list])
        return TFNNCFTensor(tf.reduce_mean(cated, axis=axis))

    @staticmethod
    def isclose_all(tensor1: TFNNCFTensor, tensor2: TFNNCFTensor, rtol=1e-05, atol=1e-08) -> bool:
        return tf.experimental.numpy.isclose(tensor1, tensor2, rtol=rtol, atol=atol)

    @staticmethod
    def stack(tensor_list: List[TFNNCFTensor]) -> TFNNCFTensor:
        return TFNNCFTensor(tf.stack([t.tensor for t in tensor_list]))

    @staticmethod
    def count_nonzero(tensor: TFNNCFTensor) -> TFNNCFTensor:
        return TFNNCFTensor(tf.math.count_nonzero(tensor.tensor))

    @staticmethod
    def abs(tensor: TFNNCFTensor) -> TFNNCFTensor:
        return TFNNCFTensor(tf.math.abs(tensor.tensor))

    @staticmethod
    def min(tensor: TFNNCFTensor, axis: int = None) -> TFNNCFTensor:
        return TFNNCFTensor(tf.reduce_min(tensor.tensor, axis=axis))

    @staticmethod
    def max(tensor: TFNNCFTensor, axis: int = None) -> TFNNCFTensor:
        return TFNNCFTensor(tf.reduce_max(tensor.tensor, axis=axis))

    @staticmethod
    def min_of_list(tensor_list: List[TFNNCFTensor], axis: int = None) -> TFNNCFTensor:
        cated = tf.concat([t.tensor for t in tensor_list])
        return TFNNCFTensor(tf.reduce_min(cated, axis=axis))

    @staticmethod
    def max_of_list(tensor_list: List[TFNNCFTensor], axis: int = None) -> TFNNCFTensor:
        cated = tf.concat([t.tensor for t in tensor_list])
        return TFNNCFTensor(tf.reduce_max(cated, axis=axis))

    @staticmethod
    def expand_dims(tensor: TFNNCFTensor, axes: List[int]) -> TFNNCFTensor:
        assert axes
        sorted_axes = sorted(axes)
        tmp = tensor.tensor
        for ax in reversed(sorted_axes):
            tmp = tf.expand_dims(tmp, axis=ax)
        return TFNNCFTensor(tmp)

    @staticmethod
    def sum(tensor: TFNNCFTensor, axes: List[int]) -> TFNNCFTensor:
        return TFNNCFTensor(tf.reduce_sum(tensor.tensor, axis=axes))

    @staticmethod
    def transpose(tensor: TFNNCFTensor, axes: List[int]) -> TFNNCFTensor:
        return TFNNCFTensor(tf.transpose(tensor.tensor, perm=axes))

    @staticmethod
    def eps(dtype: TensorDtype) -> float:
        return tf.keras.backend.epsilon()

    @staticmethod
    def median(tensor: TFNNCFTensor) -> TFNNCFTensor:
        return tensor.median()

    @staticmethod
    def clip(tensor: TFNNCFTensor, min_val: float, max_val: Optional[float] = None) -> TFNNCFTensor:
        return TFNNCFTensor(tf.clip_by_value(tensor.tensor, clip_value_min=min_val, clip_value_max=max_val))

    @staticmethod
    def ones(shape: Union[int, List[int]], dtype: TensorDtype) -> TFNNCFTensor:
        return TFNNCFTensor(tf.ones(shape, _DTYPE_MAP[dtype]))

    @staticmethod
    def squeeze(tensor: TFNNCFTensor) -> TFNNCFTensor:
        return TFNNCFTensor(tf.squeeze(tensor.tensor))

    @staticmethod
    def power(tensor: TFNNCFTensor, pwr: float) -> TFNNCFTensor:
        return TFNNCFTensor(tf.pow(tensor.tensor, pwr))

    @staticmethod
    def quantile(tensor: TFNNCFTensor, quantile: Union[float, List[float]], axis: Union[int, List[int]] = None,
                 keepdims: bool = False) -> Union[float, TFNNCFTensor]:
        np_ndarr: np.ndarray = tensor.tensor.numpy()
        return TFNNCFTensor(tf.convert_to_tensor(np.quantile(np_ndarr, q=quantile, axis=axis, keepdims=keepdims)))

    @staticmethod
    def logical_or(tensor1: TFNNCFTensor, tensor2: TFNNCFTensor) -> TFNNCFTensor:
        return TFNNCFTensor(tf.math.logical_or(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def masked_mean(tensor: TFNNCFTensor, mask: TFNNCFTensor, axis: int = None, keepdims: bool = False) -> TFNNCFTensor:
        def mean(row: tf.Tensor):
            tmp_mask = tf.not_equal(row, mask.tensor)
            filtered = tf.boolean_mask(row, tmp_mask)
            return tf.reduce_mean(filtered)
        return TFNNCFTensor(tf.map_fn(mean, tensor.tensor))

    @staticmethod
    def masked_median(tensor: TFNNCFTensor, mask: TFNNCFTensor, axis: int = None, keepdims: bool = False) -> TFNNCFTensor:
        masked_x = np.ma.array(tensor.tensor.numpy(), mask=mask.tensor.numpy())
        result = np.ma.median(masked_x, axis=axis, keepdims=False)
        if isinstance(result, (MaskedConstant, MaskedArray)):
            result = result.data
        return TFNNCFTensor(tf.convert_to_tensor(result))

    @staticmethod
    def concatenate(tensor_list: List[TFNNCFTensor], axis: int = None) -> TFNNCFTensor:
        return tf.concat([t.tensor for t in tensor_list])

    @staticmethod
    def amin(tensor: TFNNCFTensor, axis: List[int], keepdims: bool = None) -> TFNNCFTensor:
        return TFNNCFTensor(tf.experimental.numpy.amin(tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def amax(tensor: TFNNCFTensor, axis: List[int], keepdims: bool = None) -> TFNNCFTensor:
        return TFNNCFTensor(tf.experimental.numpy.amax(tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def minimum(tensor1: TFNNCFTensor, tensor2: TFNNCFTensor) -> TFNNCFTensor:
        return TFNNCFTensor(tf.minimum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def maximum(tensor1: TFNNCFTensor, tensor2: TFNNCFTensor) -> TFNNCFTensor:
        return TFNNCFTensor(tf.maximum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def unstack(tensor: TFNNCFTensor, axis: int = 0) -> List[TFNNCFTensor]:
        return [TFNNCFTensor(t) for t in tf.unstack(tensor.tensor, axis=axis)]

