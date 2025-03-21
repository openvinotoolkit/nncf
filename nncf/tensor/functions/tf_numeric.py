# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from nncf.tensor import TensorDataType
from nncf.tensor import TensorDeviceType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric as numeric
from nncf.tensor.tensor import TTensor

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


def convert_to_tf_device(device: Optional[TensorDeviceType]) -> Optional[str]:
    return DEVICE_MAP[device] if device is not None else None


def convert_to_tf_dtype(dtype: Optional[TensorDataType]) -> Optional[tf.DType]:
    return DTYPE_MAP[dtype] if dtype is not None else None


@numeric.device.register
def _(a: tf.Tensor) -> TensorDeviceType:
    if "CPU" in a.device:
        return DEVICE_MAP_REV["CPU"]
    if "GPU" in a.device:
        return DEVICE_MAP_REV["GPU"]
    return TensorDeviceType.CPU


@numeric.backend.register
def _(a: tf.Tensor) -> TensorBackend:
    return TensorBackend.tf


@numeric.squeeze.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        return tf.squeeze(a, axis)


@numeric.flatten.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reshape(a, [-1])


@numeric.max.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reduce_max(a, axis=axis, keepdims=keepdims)


@numeric.min.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reduce_min(a, axis=axis, keepdims=keepdims)


@numeric.abs.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.abs(a)


@numeric.astype.register
def _(a: tf.Tensor, dtype: TensorDataType) -> tf.Tensor:
    with tf.device(a.device):
        return tf.cast(a, DTYPE_MAP[dtype])


@numeric.dtype.register
def _(a: tf.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.dtype]


@numeric.reshape.register
def reshape(a: tf.Tensor, shape: Union[int, Tuple[int, ...]]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reshape(a, shape)


@numeric.all.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.reduce_all(a)
        return tf.reduce_all(a, axis=axis)


@numeric.allclose.register
def _(
    a: tf.Tensor, b: Union[tf.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    with tf.device(a.device):
        return bool(tf.experimental.numpy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


@numeric.any.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            return tf.reduce_any(a)
        return tf.reduce_any(a, axis=axis)


@numeric.count_nonzero.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> tf.Tensor:
    with tf.device(a.device):
        return tf.math.count_nonzero(a, axis=axis)


@numeric.isempty.register
def _(a: tf.Tensor) -> bool:
    return bool(tf.equal(tf.size(a), 0))


@numeric.isclose.register
def _(
    a: tf.Tensor, b: Union[tf.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> tf.Tensor:
    with tf.device(a.device):
        return tf.experimental.numpy.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


@numeric.maximum.register
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.maximum(x1, x2)


@numeric.minimum.register
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.minimum(x1, x2)


@numeric.ones_like.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.ones_like(a)


@numeric.where.register
def _(condition: tf.Tensor, x: Union[tf.Tensor, float, bool], y: Union[tf.Tensor, float, bool]) -> tf.Tensor:
    with tf.device(condition.device):
        return tf.where(condition, x, y)


@numeric.zeros_like.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.zeros_like(a)


@numeric.stack.register
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    with tf.device(x[0].device):
        return tf.stack(x, axis=axis)


@numeric.concatenate.register
def _(x: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
    with tf.device(x[0].device):
        return tf.concat(x, axis=axis)


@numeric.unstack.register
def _(x: tf.Tensor, axis: int = 0) -> List[tf.Tensor]:
    with tf.device(x.device):
        if not list(x.shape):
            tf.expand_dims(x, 0)
        return tf.unstack(x, axis=axis)


@numeric.moveaxis.register
def _(a: tf.Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.experimental.numpy.moveaxis(a, source, destination)


@numeric.mean.register
def _(
    a: tf.Tensor,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    keepdims: bool = False,
    dtype: Optional[TensorDataType] = None,
) -> tf.Tensor:
    with tf.device(a.device):
        a = tf.cast(a, DTYPE_MAP[dtype]) if dtype is not None else a
        return tf.reduce_mean(a, axis=axis, keepdims=keepdims)


@numeric.median.register
def _(
    a: tf.Tensor,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    keepdims: bool = False,
) -> tf.Tensor:
    with tf.device(a.device):
        if axis is None:
            a = tf.reshape(a, [-1])
        else:
            if isinstance(axis, int):
                axis = (axis,)
            destination_axis = tuple([-(i + 1) for i in range(len(axis))])
            a = tf.experimental.numpy.moveaxis(a, axis, destination_axis)
            last_axis = 1
            for i in range(len(axis)):
                last_axis *= a.shape[-(i + 1)]
            new_shape = a.shape[: -len(axis)] + [last_axis]
            a = tf.reshape(a, new_shape)
        k = 1 + a.shape[-1] // 2
        top_k = tf.math.top_k(a, k=k, sorted=True).values
        if a.shape[-1] % 2 == 0:
            median = (tf.gather(top_k, indices=[k - 2], axis=-1) + tf.gather(top_k, indices=[k - 1], axis=-1)) / 2
        else:
            median = tf.gather(top_k, indices=[k - 1], axis=-1)
        median = tf.squeeze(median, axis=-1)
        if keepdims and axis is not None:
            for axe in sorted(axis, key=lambda x: abs(x)):
                median = tf.expand_dims(median, axe)

        return median


@numeric.round.register
def _(a: tf.Tensor, decimals: int = 0) -> tf.Tensor:
    scale_factor = 10**decimals
    scaled_tensor = a * scale_factor
    with tf.device(a.device):
        rounded_tensor = tf.round(scaled_tensor)
        return rounded_tensor / scale_factor


@numeric.power.register
def _(a: tf.Tensor, exponent: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(a.device):
        if not isinstance(exponent, tf.Tensor):
            exponent_tensor = tf.convert_to_tensor(exponent, dtype=a.dtype)
        else:
            with tf.device(a.device):
                exponent_tensor = tf.identity(exponent)

        result = tf.pow(a, exponent_tensor)
        return tf.identity(result)


@numeric.quantile.register
def quantile(
    a: tf.Tensor,
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> tf.Tensor:
    a_np = a.numpy()
    quantile_np = np.quantile(a_np, q=q, axis=axis, keepdims=keepdims)
    with tf.device(a.device):
        return tf.constant(quantile_np)


@numeric.percentile.register
def percentile(
    a: tf.Tensor,
    q: Union[float, List[float]],
    axis: Optional[Union[Tuple[int, ...], int]],
    keepdims: bool = False,
) -> tf.Tensor:
    with tf.device(a.device):
        q = [x / 100 for x in q] if isinstance(q, (list, tuple)) else q / 100
        if isinstance(axis, list):
            axis = tuple(axis)
        return quantile(a, q=q, axis=axis, keepdims=keepdims)


@numeric._binary_op_nowarn.register
def _(a: tf.Tensor, b: Union[tf.Tensor, float], operator_fn: Callable[..., Any]) -> tf.Tensor:
    with tf.device(a.device):
        if not isinstance(b, tf.Tensor) and isinstance(b, (int, float)):
            b = tf.convert_to_tensor(b, dtype=a.dtype)
        result = operator_fn(a, b)
        return tf.identity(result)


@numeric._binary_reverse_op_nowarn.register
def _(a: tf.Tensor, b: Union[tf.Tensor, float], operator_fn: Callable[..., Any]) -> tf.Tensor:
    with tf.device(a.device):
        if not isinstance(b, tf.Tensor) and isinstance(b, (int, float)):
            b = tf.convert_to_tensor(b, dtype=a.dtype)
        result = operator_fn(b, a)
        return tf.identity(result)


@numeric.add.register
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        if not isinstance(x2, tf.Tensor):
            x2 = tf.convert_to_tensor(x2, dtype=x1.dtype)
        result = tf.add(x1, x2)
        return tf.identity(result)


@numeric.subtract.register
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        if not isinstance(x2, tf.Tensor):
            x2 = tf.convert_to_tensor(x2, dtype=x1.dtype)
        result = tf.subtract(x1, x2)
        return tf.identity(result)


@numeric.multiply.register
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        if not isinstance(x2, tf.Tensor):
            x2 = tf.convert_to_tensor(x2, dtype=x1.dtype)
        result = tf.multiply(x1, x2)
        return tf.identity(result)


@numeric.neg.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        result = tf.negative(a)
        return tf.identity(result)


@numeric.clip.register
def _(a: tf.Tensor, a_min: Union[tf.Tensor, float], a_max: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(a.device):
        return tf.clip_by_value(a, a_min, a_max)


@numeric.finfo.register
def _(a: tf.Tensor) -> TypeInfo:
    ti = tf.experimental.numpy.finfo(a.dtype)
    return TypeInfo(ti.eps, ti.max, ti.min)


@numeric.as_tensor_like.register
def _(a: tf.Tensor, data: Any) -> tf.Tensor:
    with tf.device(a.device):
        return tf.convert_to_tensor(data)


@numeric.item.register
def _(a: tf.Tensor) -> Union[int, float, bool]:
    return a.numpy().item()


@numeric.sum.register
def _(a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        return tf.reduce_sum(a, axis=axis, keepdims=keepdims)


@numeric.multiply.register
def _(x1: tf.Tensor, x2: Union[tf.Tensor, float]) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.multiply(x1, x2)


@numeric.var.register
def _(
    a: tf.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 0
) -> tf.Tensor:
    with tf.device(a.device):
        tf_var = tf.math.reduce_variance(a, axis=axis, keepdims=keepdims)
        if ddof:
            n = tf.shape(a)[axis] if axis is not None else tf.size(a)
            tf_var *= float(n) / float(n - ddof)
        return tf_var


@numeric.size.register
def _(a: tf.Tensor) -> int:
    return tf.size(a)


@numeric.matmul.register
def _(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.matmul(x1, x2)


@numeric.unsqueeze.register
def unsqueeze(a: tf.Tensor, axis: int) -> tf.Tensor:
    with tf.device(a.device):
        return tf.expand_dims(a, axis=axis)


@numeric.transpose.register
def _(a: tf.Tensor, axes: Optional[Tuple[int, ...]] = None) -> tf.Tensor:
    with tf.device(a.device):
        return tf.transpose(a, perm=axes)


@numeric.argsort.register
def argsort(a: tf.Tensor, axis: int = -1, descending: bool = False, stable: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        direction = "DESCENDING" if descending else "ASCENDING"
        return tf.argsort(a, axis=axis, direction=direction, stable=stable)


@numeric.diag.register
def _(a: tf.Tensor, k: int = 0) -> tf.Tensor:
    with tf.device(a.device):
        rank = tf.rank(a)
        if rank == 1:
            return tf.linalg.diag(a, k=k)
        elif rank == 2:
            return tf.linalg.diag_part(a, k=k)
        else:
            error_msg = "Input tensor must be 1D or 2D."
            raise ValueError(error_msg)


@numeric.logical_or.register
def _(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    with tf.device(x1.device):
        return tf.logical_or(x1, x2)


@numeric.masked_mean.register
def masked_mean(
    x: tf.Tensor, mask: Optional[tf.Tensor], axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool = False
) -> tf.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)

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


@numeric.masked_median.register
def masked_median(
    x: tf.Tensor, mask: Optional[tf.Tensor], axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool = False
) -> tf.Tensor:
    if mask is None:
        return numeric.median(x, axis=axis, keepdims=keepdims)

    if isinstance(axis, list):
        axis = tuple(axis)

    masked_x = tf.where(mask, np.nan, x)
    np_masked_x = masked_x.numpy()
    np_masked_median = np.nanquantile(np_masked_x, 0.5, axis=axis, keepdims=keepdims)

    with tf.device(x.device):
        ret = tf.constant(np_masked_median)
        ret = tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

        return ret


@numeric.expand_dims.register
def expand_dims(a: tf.Tensor, axis: Union[int, Tuple[int, ...]]) -> tf.Tensor:
    axes_tuple: Tuple[int, ...]
    if isinstance(axis, int):
        axes_tuple = (axis,)
    elif isinstance(axis, tuple):
        axes_tuple = axis
    else:
        error_msg = f"axis must be int or tuple, got {type(axis)}"
        raise TypeError(error_msg)

    if len(set(axes_tuple)) != len(axes_tuple):
        error_msg = "repeated axis"
        raise ValueError(error_msg)

    out_ndim = len(axes_tuple) + a.ndim

    norm_axis = []
    for ax in axes_tuple:
        if ax < -out_ndim or ax >= out_ndim:
            error_msg = f"axis {ax} is out of bounds for array of dimension {out_ndim}"
            raise ValueError(error_msg)
        norm_axis.append(ax + out_ndim if ax < 0 else ax)

    shape_it = iter(a.shape)
    shape = [1 if ax in norm_axis else next(shape_it) for ax in range(out_ndim)]
    return tf.reshape(a, shape)


@numeric.clone.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.identity(a)


@numeric.searchsorted.register
def searchsorted(
    a: tf.Tensor, v: tf.Tensor, side: Literal["left", "right"] = "left", sorter: Optional[tf.Tensor] = None
) -> tf.Tensor:
    if side not in ["right", "left"]:
        error_msg = f"Invalid value for 'side': {side}. Expected 'right' or 'left'."
        raise ValueError(error_msg)
    if a.ndim != 1:
        error_msg = f"Input tensor 'a' must be 1-D. Received {a.ndim}-D tensor."
        raise ValueError(error_msg)
    sorted_a = tf.sort(a)
    return tf.searchsorted(sorted_sequence=sorted_a, values=v, side=side)


def zeros(
    shape: Tuple[int, ...],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    tf_dtype = DTYPE_MAP[dtype] if dtype is not None else None
    tf_device = DEVICE_MAP[device] if device is not None else None
    with tf.device(tf_device):
        return tf.zeros(shape, dtype=tf_dtype)


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    tf_dtype = DTYPE_MAP[dtype] if dtype is not None else None
    tf_device = DEVICE_MAP[device] if device is not None else None
    p_args = (n,) if m is None else (n, m)
    with tf.device(tf_device):
        return tf.eye(*p_args, dtype=tf_dtype)


def arange(
    start: float,
    end: float,
    step: float,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    tf_dtype = DTYPE_MAP[dtype] if dtype is not None else None
    tf_device = DEVICE_MAP[device] if device is not None else None
    with tf.device(tf_device):
        return tf.range(start, end, step, dtype=tf_dtype)


def from_numpy(ndarray: npt.NDArray[Any]) -> tf.Tensor:
    with tf.device("CPU"):
        return tf.constant(ndarray)


@numeric.as_numpy_tensor.register
def as_numpy_tensor(a: tf.Tensor) -> npt.NDArray[Any]:
    return a.numpy()


@numeric.log2.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.math.log(a) / tf.math.log(2.0)


@numeric.ceil.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.math.ceil(a)


def tensor(
    data: Union[TTensor, Sequence[float]],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> tf.Tensor:
    tf_device = convert_to_tf_device(device)
    tf_dtype = convert_to_tf_dtype(dtype)
    with tf.device(tf_device):
        return tf.constant(data, dtype=tf_dtype)
