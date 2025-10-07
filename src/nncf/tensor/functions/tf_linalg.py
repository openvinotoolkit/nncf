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

import warnings
from typing import Literal, Optional, Union

import tensorflow as tf

from nncf.tensor.functions import linalg


@linalg.norm.register
def _(
    a: tf.Tensor,
    ord: Union[Literal["fro", "nuc"], float, None] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> tf.Tensor:
    rank = tf.rank(a)

    if ord is None:
        if axis is None and rank == 2:
            ord = "fro"
        else:
            ord = 2

    if rank == 2 and axis is None:
        axis = (0, 1)

    with tf.device(a.device):
        if ord == "nuc" and isinstance(axis, tuple) and len(axis) == 2:
            if rank == 2:
                s = tf.linalg.svd(a, compute_uv=False)
                result = tf.reduce_sum(s, axis=-1)
                if keepdims:
                    result_shape = [1 if i in axis else dim for i, dim in enumerate(a.shape)]
                    result = tf.reshape(result, result_shape)
                return result
            else:
                perm = list(range(rank))
                for i in sorted(axis, reverse=True):
                    perm.pop(i)
                perm = perm + list(axis)

                a_transposed = tf.transpose(a, perm=perm)

                batch_shape = a_transposed.shape[:-2]
                matrix_shape = a_transposed.shape[-2:]
                a_reshaped = tf.reshape(a_transposed, [-1, matrix_shape[0], matrix_shape[1]])

                s = tf.linalg.svd(a_reshaped, compute_uv=False)

                result = tf.reduce_sum(s, axis=-1)

                result = tf.reshape(result, batch_shape)

                if keepdims:
                    for ax in sorted(axis):
                        result = tf.expand_dims(result, ax)

                return result

        if ord == 0:
            return tf.cast(tf.math.count_nonzero(a, axis=axis, keepdims=keepdims), a.dtype)

        if ord == -1 and isinstance(axis, tuple) and len(axis) != 1:
            if rank != 2:
                msg = "ord=-1 is only supported for 2D tensors"
                raise ValueError(msg)
            result = tf.reduce_min(tf.reduce_sum(tf.abs(a), axis=axis[0]), keepdims=keepdims)
            if keepdims:
                result = tf.reshape(result, [1, 1])
            return result

        if ord == 1 and isinstance(axis, tuple) and len(axis) != 1:
            if rank != 2:
                msg = "ord=1 is only supported for 2D tensors"
                raise ValueError(msg)
            result = tf.reduce_max(tf.reduce_sum(tf.abs(a), axis=axis[0]), keepdims=keepdims)
            if keepdims:
                result = tf.reshape(result, [1, 1])
            return result

        if ord == -2 and isinstance(axis, tuple) and len(axis) != 1:
            if rank != 2:
                msg = "ord=-2 is only supported for 2D tensors"
                raise ValueError(msg)
            s = tf.linalg.svd(a, compute_uv=False)
            result = tf.reduce_min(s, axis=-1)
            if keepdims:
                result = tf.reshape(result, [1, 1])
            return result

        if ord == 2 and isinstance(axis, tuple) and len(axis) != 1:
            if rank != 2:
                msg = "ord=2 is only supported for 2D tensors"
                raise ValueError(msg)
            s = tf.linalg.svd(a, compute_uv=False)
            result = tf.reduce_max(s, axis=-1)
            if keepdims:
                result = tf.reshape(result, [1, 1])
            return result

        if ord == float("inf") and isinstance(axis, tuple) and len(axis) != 1:
            if rank != 2:
                msg = "ord=inf is only supported for 2D tensors"
                raise ValueError(msg)
            result = tf.reduce_max(tf.reduce_sum(tf.abs(a), axis=axis[1]), keepdims=keepdims)
            if keepdims:
                result = tf.reshape(result, [1, 1])
            return result

        if ord == -float("inf") and isinstance(axis, tuple) and len(axis) != 1:
            if rank != 2:
                msg = "ord=-inf is only supported for 2D tensors"
                raise ValueError(msg)
            result = tf.reduce_min(tf.reduce_sum(tf.abs(a), axis=axis[1]), keepdims=keepdims)
            if keepdims:
                result = tf.reshape(result, [1, 1])
            return result

        try:
            return tf.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
        except (TypeError, ValueError) as exc:
            if axis is not None:
                if ord == 2:
                    squared = tf.square(a)
                    sum_squares = tf.reduce_sum(squared, axis=axis, keepdims=keepdims)
                    return tf.sqrt(sum_squares)
                elif ord == 1:
                    return tf.reduce_sum(tf.abs(a), axis=axis, keepdims=keepdims)
                elif ord == float("inf"):
                    return tf.reduce_max(tf.abs(a), axis=axis, keepdims=keepdims)
                elif ord == -float("inf"):
                    return tf.reduce_min(tf.abs(a), axis=axis, keepdims=keepdims)

            msg = f"Unsupported combination of ord={ord} and axis={axis}"
            raise ValueError(msg) from exc


@linalg.cholesky.register
def _(a: tf.Tensor, upper: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        cholesky = tf.linalg.cholesky(a)
        if upper:
            perm = list(range(tf.rank(a)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            cholesky = tf.transpose(cholesky, perm=perm)
        return cholesky


@linalg.cholesky_inverse.register
def _(a: tf.Tensor, upper: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        if upper:
            perm = list(range(tf.rank(a)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            a = tf.transpose(a, perm=perm)

        eye = tf.eye(a.shape[0], dtype=a.dtype)
        return tf.linalg.cholesky_solve(a, eye)


@linalg.inv.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.linalg.inv(a)


@linalg.pinv.register
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.linalg.pinv(a)


@linalg.lstsq.register
def _(a: tf.Tensor, b: tf.Tensor, driver: Optional[str] = None) -> tf.Tensor:
    with tf.device(a.device):
        if driver is not None:
            warnings.warn("Driver specifying is not supported in TensorFlow lstsq method")  # noqa: B028
        if tf.rank(b) == 1:
            b = tf.expand_dims(b, axis=1)

        return tf.linalg.lstsq(a, b)


@linalg.svd.register
def _(a: tf.Tensor, full_matrices: Optional[bool] = True) -> tf.Tensor:
    with tf.device(a.device):
        s, u, v = tf.linalg.svd(a, full_matrices=full_matrices)

        return u, s, tf.transpose(v, conjugate=True)
