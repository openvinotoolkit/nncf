# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from nncf.tensor import TensorDataType
from nncf.tensor import TensorDeviceType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric as numeric

DTYPE_MAP = {
    TensorDataType.float16: tf.float16,
    TensorDataType.bfloat16: tf.bfloat16,
    TensorDataType.float32: tf.float32,
    TensorDataType.float64: tf.float64,
    TensorDataType.int8: tf.int8,
    TensorDataType.int32: tf.int32,
    TensorDataType.int64: tf.int64,
    TensorDataType.uint8: tf.uint8,
}

DEVICE_MAP = {TensorDeviceType.CPU: "CPU", TensorDeviceType.GPU: "GPU"}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}
DEVICE_MAP_REV = {v: k for k, v in DEVICE_MAP.items()}


@numeric.device.register(tf.Tensor)
def _(a: tf.Tensor) -> TensorDeviceType:
    if "CPU" in a.device:
        return DEVICE_MAP_REV["CPU"]
    if "GPU" in a.device:
        return DEVICE_MAP_REV["GPU"]


@numeric.backend.register(tf.Tensor)
def _(a: tf.Tensor) -> TensorBackend:
    return TensorBackend.tf


@numeric.squeeze.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.squeeze(a)
        if isinstance(axis, Tuple) and any(a.shape[i] != 1 for i in axis):
            raise ValueError("Cannot select an axis to squeeze out which has size not equal to one")
        return tf.squeeze(a, axis)


@numeric.flatten.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reshape(a, [-1])


@numeric.max.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.reduce_max(a)
        return tf.reduce_max(a, axis=axis, keepdims=keepdim)


@numeric.min.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.reduce_min(a)
        return tf.reduce_min(a, axis=axis, keepdims=keepdim)


@numeric.abs.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.abs(a)


@numeric.astype.register(tf.Tensor)
def _(a: tf.Tensor, dtype: TensorDataType) -> tf.Tensor:
    with tf.device(a.device):
        return tf.cast(a, DTYPE_MAP[dtype])


@numeric.dtype.register(tf.Tensor)
def _(a: tf.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.dtype]


@numeric.reshape.register(tf.Tensor)
def _(a: tf.Tensor, shape: Tuple[int, ...]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reshape(a, shape)


@numeric.all.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.reduce_all(a)
        return tf.reduce_all(a, axis=axis)


@numeric.allclose.register(tf.Tensor)
def _(
    a: tf.Tensor, b: Union[tf.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    with tf.device(a.device):
        if not isinstance(b, tf.Tensor):
            b = tf.constant(b)
        return tf.experimental.numpy.allclose(a, tf.cast(b, a.dtype), rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.any.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.reduce_any(a)
        return tf.reduce_any(a, axis=axis)


@numeric.count_nonzero.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        return tf.math.count_nonzero(a, axis=axis)


@numeric.isempty.register(tf.Tensor)
def _(a: tf.Tensor) -> bool:
    return bool(tf.equal(tf.size(a), 0).numpy())


@numeric.isclose.register(tf.Tensor)
def _(
    a: tf.Tensor, b: Union[tf.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> tf.Tensor:
    with tf.device(a.device):
        return tf.experimental.numpy.isclose(a, tf.cast(b, a.dtype), atol=atol, rtol=rtol, equal_nan=equal_nan)


@numeric.maximum.register(tf.Tensor)
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.maximum(x1, x2)


@numeric.minimum.register(tf.Tensor)
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.minimum(x1, x2)


@numeric.ones_like.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.ones_like(a)


@numeric.where.register(tf.Tensor)
def _(condition: tf.Tensor, x: Union[tf.Tensor, float, bool], y: Union[tf.Tensor, float, bool]) -> tf.Tensor:
    with tf.device(condition.device):
        return tf.where(condition, x, y)


@numeric.zeros_like.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.zeros_like(a)


@numeric.stack.register(tf.Tensor)
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    with tf.device(x[0].device):
        return tf.stack(x, axis=axis)


@numeric.concatenate.register(tf.Tensor)
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    with tf.device(x[0].device):
        return tf.concat(x, axis=axis)


@numeric.unstack.register(tf.Tensor)
def _(x: tf.Tensor, axis: int = 0) -> List[tf.Tensor]:
    with tf.device(x.device):
        if not list(x.shape):
            tf.expand_dims(x, 0)
        return tf.unstack(x, axis=axis)


@numeric.moveaxis.register(tf.Tensor)
def _(a: tf.Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.experimental.numpy.moveaxis(a, source, destination)


@numeric.mean.register(tf.Tensor)
def _(
    a: tf.Tensor,
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
    dtype: Optional[TensorDataType] = None,
) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reduce_mean(a, axis=axis, keepdims=keepdims)


@numeric.median.register(tf.Tensor)
def _(
    a: tf.Tensor,
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> tf.Tensor:
    numpy_a = np.array(a)
    numpy_median = np.median(numpy_a, axis=axis, keepdims=keepdims)

    with tf.device(a.device):
        tf_median = tf.constant(numpy_median)

    return tf_median


@numeric.round.register(tf.Tensor)
def _(a: tf.Tensor, decimals=0) -> tf.Tensor:
    scale_factor = 10**decimals
    scaled_tensor = a * scale_factor
    with tf.device(a.device):
        rounded_tensor = tf.round(scaled_tensor)
        return rounded_tensor / scale_factor


@numeric.power.register(tf.Tensor)
def _(a: tf.Tensor, exponent: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.pow(a, exponent)


@numeric.quantile.register(tf.Tensor)
def quantile(
    a: tf.Tensor,
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
) -> tf.Tensor:
    a_np = a.numpy()
    quantile_np = np.quantile(a_np, q=q, axis=axis, keepdims=keepdims)
    with tf.device(a.device):
        return tf.constant(quantile_np)


@numeric.percentile.register(tf.Tensor)
def _(
    a: tf.Tensor,
    q: Union[float, List[float]],
    axis: Union[int, Tuple[int, ...], List[int]],
    keepdims: bool = False,
) -> List[Union[tf.Tensor, np.generic]]:
    with tf.device(a.device):
        q = [x / 100 for x in q] if isinstance(q, (list, tuple)) else q / 100
        return numeric.quantile(a, q=q, axis=axis, keepdims=keepdims)


@numeric._binary_op_nowarn.register(tf.Tensor)
def _(a: tf.Tensor, b: Union[tf.Tensor, float], operator_fn: Callable) -> tf.Tensor:
    with tf.device(a.device):
        return operator_fn(a, b)


@numeric._binary_reverse_op_nowarn.register(tf.Tensor)
def _(a: tf.Tensor, b: Union[tf.Tensor, float], operator_fn: Callable) -> tf.Tensor:
    with tf.device(a.device):
        return operator_fn(b, a)


@numeric.clip.register(tf.Tensor)
def _(a: tf.Tensor, a_min: Union[tf.Tensor, float], a_max: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.clip_by_value(a, a_min, a_max)


@numeric.finfo.register(tf.Tensor)
def _(a: tf.Tensor) -> TypeInfo:
    ti = tf.experimental.numpy.finfo(a.dtype)
    return TypeInfo(ti.eps, ti.max, ti.min)


@numeric.as_tensor_like.register(tf.Tensor)
def _(a: tf.Tensor, data: Any) -> tf.Tensor:
    with tf.device(a.device):
        return tf.convert_to_tensor(data)


@numeric.item.register(tf.Tensor)
def _(a: tf.Tensor) -> Union[int, float, bool]:
    a = tf.reshape(a, [])
    np_item = a.numpy()
    if isinstance(np_item, np.floating):
        return float(np_item)
    if isinstance(np_item, np.bool_):
        return bool(np_item)

    return int(np_item)


@numeric.sum.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reduce_sum(a, axis=axis, keepdims=keepdims)


@numeric.multiply.register(tf.Tensor)
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.multiply(x1, x2)


@numeric.var.register(tf.Tensor)
def _(
    a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 0
) -> tf.Tensor:
    with tf.device(a.device):
        tf_var = tf.math.reduce_variance(a, axis=axis, keepdims=keepdims)
        if ddof:
            n = tf.shape(a)[axis] if axis is not None else tf.size(a)
            tf_var *= float(n) / float(n - ddof)
        return tf_var


@numeric.size.register(tf.Tensor)
def _(a: tf.Tensor) -> int:
    return tf.size(a)


@numeric.matmul.register(tf.Tensor)
def _(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.matmul(x1, x2)


@numeric.unsqueeze.register(tf.Tensor)
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        return tf.expand_dims(a, axis=axis)


@numeric.transpose.register(tf.Tensor)
def _(a: tf.Tensor, axes: Optional[Tuple[int, ...]] = None) -> tf.Tensor:
    with tf.device(a.device):
        return tf.transpose(a, perm=axes)


@numeric.argsort.register(tf.Tensor)
def _(a: tf.Tensor, axis: int = -1, descending=False, stable=False) -> tf.Tensor:
    with tf.device(a.device):
        direction = "DESCENDING" if descending else "ASCENDING"
        return tf.argsort(a, axis=axis, direction=direction, stable=stable)


@numeric.diag.register(tf.Tensor)
def _(a: tf.Tensor, k: int = 0) -> tf.Tensor:
    with tf.device(a.device):
        if a._rank() == 2:
            if k == 0:
                return tf.linalg.diag_part(a)
            elif k > 0:
                return tf.linalg.diag_part(a[:, k:])
            else:
                return tf.linalg.diag_part(a[-k:, :])

        if a._rank() == 1:
            return tf.linalg.diag(a, k=k)


@numeric.logical_or.register(tf.Tensor)
def _(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.logical_or(x1, x2)


@numeric.masked_mean.register(tf.Tensor)
def _(
    x: tf.Tensor, mask: Optional[tf.Tensor], axis: Union[int, Tuple[int, ...], List[int]], keepdims=False
) -> tf.Tensor:
    with tf.device(x.device):
        if mask is None:
            return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
        flipped_mask = ~mask
        valid_counts = tf.reduce_sum(tf.cast(flipped_mask, x.dtype), axis=axis, keepdims=keepdims)
        masked_x = tf.where(mask, tf.zeros_like(x), x)
        valid_sum = tf.reduce_sum(masked_x, axis=axis, keepdims=keepdims)

        ret = valid_sum / valid_counts
        ret = tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

        return ret


@numeric.masked_median.register(tf.Tensor)
def _(
    x: tf.Tensor, mask: Optional[tf.Tensor], axis: Union[int, Tuple[int, ...], List[int]], keepdims=False
) -> tf.Tensor:
    if mask is None:
        return numeric.median(x, axis=axis, keepdims=keepdims)

    masked_x = tf.where(mask, np.nan, x)
    np_masked_x = masked_x.numpy()
    np_masked_median = np.nanquantile(np_masked_x, 0.5, axis=axis, keepdims=keepdims)

    with tf.device(x.device):
        ret = tf.constant(np_masked_median)
        ret = tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

        return ret


@numeric.expand_dims.register(tf.Tensor)
def _(a: tf.Tensor, axis: Union[int, Tuple[int, ...], List[int]]) -> np.ndarray:
    if type(axis) not in (tuple, list):
        axis = (axis,)

    if len(set(axis)) != len(axis):
        raise ValueError("repeated axis")

    out_ndim = len(axis) + a.ndim

    norm_axis = []
    for ax in axis:
        if ax < -out_ndim or ax >= out_ndim:
            raise ValueError(f"axis {ax} is out of bounds for array of dimension {out_ndim}")
        norm_axis.append(ax + out_ndim if ax < 0 else ax)

    shape_it = iter(a.shape)
    shape = [1 if ax in norm_axis else next(shape_it) for ax in range(out_ndim)]
    return tf.reshape(a, shape)


@numeric.clone.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.identity(a)


@numeric.searchsorted.register(tf.Tensor)
def _(a: tf.Tensor, v: tf.Tensor, side: str = "left", sorter: Optional[tf.Tensor] = None) -> tf.Tensor:
    if side not in ["right", "left"]:
        raise ValueError(f"Invalid value for 'side': {side}. Expected 'right' or 'left'.")
    if a.ndim != 1:
        raise ValueError(f"Input tensor 'a' must be 1-D. Received {a.ndim}-D tensor.")
    sorted_a = tf.sort(a)
    return tf.searchsorted(sorted_sequence=sorted_a, values=v, side=side)


def zeros(
    shape: Tuple[int, ...],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    if dtype is not None:
        dtype = DTYPE_MAP[dtype]
    if device is not None:
        device = DEVICE_MAP[device]
    with tf.device(device):
        return tf.zeros(shape, dtype=dtype)


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    if dtype is not None:
        dtype = DTYPE_MAP[dtype]
    if device is not None:
        device = DEVICE_MAP[device]
    p_args = (n,) if m is None else (n, m)
    with tf.device(device):
        return tf.eye(*p_args, dtype=dtype)


def arange(
    start: float,
    end: float,
    step: float,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    if dtype is not None:
        dtype = DTYPE_MAP[dtype]
    if device is not None:
        device = DEVICE_MAP[device]
    with tf.device(device):
        return tf.range(start, end, step, dtype=dtype)


def from_numpy(ndarray: np.ndarray) -> tf.Tensor:
    with tf.device("CPU"):
        return tf.constant(ndarray)


@numeric.log2.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.math.log(a) / tf.math.log(2.0)


@numeric.ceil.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.math.ceil(a)
