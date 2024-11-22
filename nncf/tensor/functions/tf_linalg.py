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

import warnings
from typing import Optional, Tuple, Union

import tensorflow as tf

from nncf.tensor.functions import linalg


@linalg.norm.register(tf.Tensor)
def _(
    a: tf.Tensor,
    ord: Optional[Union[str, float, int]] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> tf.Tensor:
    if axis is None:
        axis = 0 if a._rank() == 1 else (0, 1)

    if ord is None or (a._rank() == 1 and ord == "fro"):
        ord = "euclidean"

    with tf.device(a.device):
        if ord == "nuc":
            s, _, _ = tf.linalg.svd(a)
            return tf.reduce_sum(s)

        return tf.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)


@linalg.cholesky.register(tf.Tensor)
def _(a: tf.Tensor, upper: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        cholesky = tf.linalg.cholesky(a)
        if upper:
            perm = list(range(tf.rank(a)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            cholesky = tf.transpose(cholesky, perm=perm)
        return cholesky


@linalg.cholesky_inverse.register(tf.Tensor)
def _(a: tf.Tensor, upper: bool = False) -> tf.Tensor:
    with tf.device(a.device):
        if upper:
            perm = list(range(tf.rank(a)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            a = tf.transpose(a, perm=perm)

        eye = tf.eye(a.shape[0], dtype=a.dtype)
        return tf.linalg.cholesky_solve(a, eye)


@linalg.inv.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.linalg.inv(a)


@linalg.pinv.register(tf.Tensor)
def _(a: tf.Tensor) -> tf.Tensor:
    with tf.device(a.device):
        return tf.linalg.pinv(a)


@linalg.lstsq.register(tf.Tensor)
def _(a: tf.Tensor, b: tf.Tensor, driver: Optional[str] = None) -> tf.Tensor:
    with tf.device(a.device):
        if driver is not None:
            warnings.warn("Driver specifying is not supported in TensorFlow lstsq method")
        if tf.rank(b) == 1:
            b = tf.expand_dims(b, axis=0)
        perm = list(range(tf.rank(b)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        b = tf.transpose(b, perm=perm)

        return tf.linalg.lstsq(a, b)


@linalg.svd.register(tf.Tensor)
def _(a: tf.Tensor, full_matrices: Optional[bool] = True) -> tf.Tensor:
    with tf.device(a.device):
        s, u, v = tf.linalg.svd(a, full_matrices=full_matrices)

        return u, s, tf.transpose(v)
