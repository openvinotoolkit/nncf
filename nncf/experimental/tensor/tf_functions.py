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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from nncf.experimental.tensor import TensorDataType
from nncf.experimental.tensor import TensorDeviceType
from nncf.experimental.tensor import functions as fns

DTYPE_MAP: Dict[TensorDataType, tf.DType] = {
    TensorDataType.float32: tf.float32,
    TensorDataType.int64: tf.int64,
}

_INV_DTYPE_MAP: Dict[tf.DType, TensorDataType] = {v: k for k, v in DTYPE_MAP.items()}


def register_tf_types(singledispatch_fn):
    """
    Decorator to register function to singledispatch for numpy classes.

    :param singledispatch_fn: singledispatch function.
    """

    def inner(func):
        singledispatch_fn.register(tf.Tensor)(func)
        singledispatch_fn.register(EagerTensor)(func)
        singledispatch_fn.register(ResourceVariable)(func)
        return func

    return inner


@register_tf_types(fns.device)
def _(a: tf.Tensor) -> TensorDeviceType:
    return a.device


@register_tf_types(fns.squeeze)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> tf.Tensor:
    return tf.squeeze(a, axis=axis)


@register_tf_types(fns.flatten)
def _(a: tf.Tensor) -> tf.Tensor:
    return tf.reshape(a, [-1])


@register_tf_types(fns.max)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> tf.Tensor:
    return tf.reduce_max(a, axis=axis)


@register_tf_types(fns.min)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> tf.Tensor:
    return tf.reduce_min(a, axis=axis)


@register_tf_types(fns.abs)
def _(a: tf.Tensor) -> tf.Tensor:
    return tf.math.abs(a)


@register_tf_types(fns.astype)
def _(a: tf.Tensor, dtype: TensorDataType) -> tf.Tensor:
    return tf.cast(a, DTYPE_MAP[dtype])


@register_tf_types(fns.dtype)
def _(a: tf.Tensor) -> TensorDataType:
    return _INV_DTYPE_MAP[a.dtype]


@register_tf_types(fns.reshape)
def _(a: tf.Tensor, shape: List[int]) -> tf.Tensor:
    return tf.reshape(a, shape)


@register_tf_types(fns.all)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Union[tf.Tensor, bool]:
    return tf.reduce_all(a, axis=axis)


@register_tf_types(fns.allclose)
def _(a: tf.Tensor, b: tf.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    return tf.experimental.numpy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@register_tf_types(fns.any)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Union[tf.Tensor, bool]:
    return tf.reduce_any(a, axis=axis)


@register_tf_types(fns.count_nonzero)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> tf.Tensor:
    return tf.math.count_nonzero(a, axis=axis)


@register_tf_types(fns.isempty)
def _(a: tf.Tensor) -> bool:
    return int(tf.size(a)) == 0


@register_tf_types(fns.isclose)
def _(a: tf.Tensor, b: tf.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    return tf.experimental.numpy.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


@register_tf_types(fns.maximum)
def _(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return tf.maximum(x1, x2)


@register_tf_types(fns.minimum)
def _(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return tf.minimum(x1, x2)


@register_tf_types(fns.ones_like)
def _(a: tf.Tensor) -> tf.Tensor:
    return tf.ones_like(a)


@register_tf_types(fns.where)
def _(condition: tf.Tensor, x: Union[tf.Tensor, float, bool], y: Union[tf.Tensor, float, bool]) -> tf.Tensor:
    return tf.where(condition, x, y)


@register_tf_types(fns.zeros_like)
def _(a: tf.Tensor) -> tf.Tensor:
    return tf.zeros_like(a)


@register_tf_types(fns.stack)
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    return tf.stack(x, axis=axis)


@register_tf_types(fns.unstack)
def _(x: tf.Tensor, axis: int = 0) -> List[tf.Tensor]:
    return tf.unstack(x, axis=axis)


@register_tf_types(fns.moveaxis)
def _(a: tf.Tensor, source: Union[int, List[int]], destination: Union[int, List[int]]) -> tf.Tensor:
    return tf.experimental.numpy.moveaxis(a, source, destination)


@register_tf_types(fns.mean)
def _(a: tf.Tensor, axis: Union[int, List[int]] = None, keepdims: Optional[bool] = None) -> tf.Tensor:
    return tf.reduce_mean(a, axis=axis, keepdims=keepdims)


@register_tf_types(fns.round)
def _(a: tf.Tensor, decimals=0) -> tf.Tensor:
    return tf.round(a)


@register_tf_types(fns.binary_operator)
def _(a: tf.Tensor, b: tf.Tensor, operator_fn: Callable) -> tf.Tensor:
    return operator_fn(a, b)


@register_tf_types(fns.binary_reverse_operator)
def _(a: tf.Tensor, b: tf.Tensor, operator_fn: Callable) -> tf.Tensor:
    return operator_fn(b, a)


@register_tf_types(fns.to_numpy)
def _(a: tf.Tensor) -> np.ndarray:
    return a.numpy()


@register_tf_types(fns.inf)
def _(a: tf.Tensor) -> Any:
    return np.inf


def _expand_scalars_in_tensor_list(tensor_list: List[tf.Tensor]) -> List[tf.Tensor]:
    retval = []
    for t in tensor_list:
        if len(t.size()) == 0:
            retval.append(t.reshape(1))
        else:
            retval.append(t)
    return retval


@register_tf_types(fns.concatenate)
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    return tf.concat(x, axis)


@register_tf_types(fns.min_of_list)
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    cated = tf.concat(x, axis=-1)
    return tf.reduce_min(cated, axis=axis)


@register_tf_types(fns.max_of_list)
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    cated = tf.concat(x, axis=-1)
    return tf.reduce_max(cated, axis=axis)


@register_tf_types(fns.amax)
def _(a: tf.Tensor, axis: Optional[List[int]] = None, keepdims: Optional[bool] = None) -> tf.Tensor:
    return tf.experimental.numpy.amax(a, axis=axis, keepdims=keepdims)


@register_tf_types(fns.amin)
def _(a: tf.Tensor, axis: Optional[List[int]] = None, keepdims: Optional[bool] = None) -> tf.Tensor:
    return tf.experimental.numpy.amin(a, axis=axis, keepdims=keepdims)


@register_tf_types(fns.clip)
def _(a: tf.Tensor, min_val: float, max_val: Optional[float] = None) -> tf.Tensor:
    return tf.clip_by_value(a, clip_value_min=min_val, clip_value_max=max_val)


@register_tf_types(fns.sum)
def _(a: tf.Tensor, axes: List[int]) -> tf.Tensor:
    return tf.reduce_sum(a, axis=axes)


@register_tf_types(fns.transpose)
def _(a: tf.Tensor, axes: List[int]) -> tf.Tensor:
    return tf.transpose(a, perm=axes)


@register_tf_types(fns.eps)
def _(a: tf.Tensor, dtype: TensorDataType) -> float:
    return tf.keras.backend.epsilon()


@register_tf_types(fns.median)
def _(a: tf.Tensor, axis: Union[int, Tuple[int]] = None, keepdims: Optional[bool] = None) -> tf.Tensor:
    # TODO (vshampor): looks like calculating median in TF is only possible through an extra dependency
    #  TFNNCFTensorflow_probability
    numpy_array: np.ndarray = a.numpy()
    return tf.convert_to_tensor(np.median(numpy_array, axis=axis, keepdims=keepdims))


@register_tf_types(fns.power)
def _(a: tf.Tensor, pwr: float) -> tf.Tensor:
    return tf.pow(a, pwr)


@register_tf_types(fns.matmul)
def _(
    a: tf.Tensor,
    b: tf.Tensor,
) -> Union[np.ndarray, np.number]:
    return tf.matmul(a, b)


@register_tf_types(fns.quantile)
def _(
    a: tf.Tensor,
    q: Union[float, List[float]],
    axis: Union[int, List[int]] = None,
    keepdims: Optional[bool] = None,
) -> Union[float, tf.Tensor]:
    np_ndarr: np.ndarray = a.numpy()
    return tf.convert_to_tensor(np.quantile(np_ndarr, q=q, axis=axis, keepdims=keepdims))


@register_tf_types(fns.logical_or)
def _(tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
    return tf.math.logical_or(tensor1, tensor2)


@register_tf_types(fns.masked_mean)
def _(a: tf.Tensor, mask: tf.Tensor, axis: int = None, keepdims: Optional[bool] = None) -> tf.Tensor:
    def mean(row: tf.Tensor):
        tmp_mask = tf.not_equal(row, mask)
        filtered = tf.boolean_mask(row, tmp_mask)
        return tf.reduce_mean(filtered)

    return tf.map_fn(mean, a)


@register_tf_types(fns.masked_median)
def _(a: tf.Tensor, mask: tf.Tensor, axis: int = None, keepdims: Optional[bool] = None) -> tf.Tensor:
    masked_x = np.ma.array(a.numpy(), mask=mask.numpy())
    result = np.ma.median(masked_x, axis=axis, keepdims=False)
    if isinstance(result, (MaskedConstant, MaskedArray)):
        result = result.data
    return tf.convert_to_tensor(result)


@register_tf_types(fns.size)
def _(a: tf.Tensor) -> int:
    return int(tf.size(a))
