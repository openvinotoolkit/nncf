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

import functools
from typing import Optional, Tuple, Union

from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor.functions.dispatcher import tensor_guard


@functools.singledispatch
@tensor_guard
def norm(
    a: Tensor,
    ord: Optional[Union[str, float, int]] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """
    Computes a vector or matrix norm.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and otherwise raise a ValueError.

    :param a: The input tensor.
    :param ord: Order of norm. Default: None.
    :param axis: Axis over which to compute the vector or matrix norm. Default: None.
    :param keepdims: If set to True, the reduced dimensions are retained in the result
        as dimensions with size one. Default: False.
    :return: Norm of the matrix or vector.
    """
    return Tensor(norm(a.data, ord, axis, keepdims))


@functools.singledispatch
@tensor_guard
def cholesky(a: Tensor, upper: bool = False) -> Tensor:
    """
    Computes the Cholesky decomposition of a complex Hermitian or real symmetric
    positive-definite matrix `a`.

    If `upper` is False, then the Cholesky decomposition, `L * L.H` of the matrix `a`
    is returned, where `L` is lower-triangular and .H is the conjugate transpose operator
    If `upper` is True, then the conjugate transpose tensor of the tensor with upper=False
    is returned.

    :param a: The input tensor of size (n,n).
    :param upper: Whether to compute the upper- or lower-triangular Cholesky factorization.
        Default is lower-triangular.
    :return: Upper- or lower-triangular Cholesky factor of `a`.
    """
    return Tensor(cholesky(a.data, upper))


@functools.singledispatch
@tensor_guard
def cholesky_inverse(a: Tensor, upper: bool = False) -> Tensor:
    """
    Computes the inverse of a complex Hermitian or real symmetric positive-definite matrix given
    its Cholesky decomposition.

    :param a: The input tensor of size (n,n) consisting of lower or upper triangular Cholesky
        decompositions of symmetric or Hermitian positive-definite matrices.
    :param upper: The flag that indicates whether the input tensor is lower triangular or
        upper triangular. Default: False.
    :return: The inverse of matrix given its Cholesky decomposition.
    """
    return Tensor(cholesky_inverse(a.data, upper))


@functools.singledispatch
@tensor_guard
def inv(a: Tensor) -> Tensor:
    """
    Computes the inverse of a matrix.

    :param a: The input tensor of shape (*, n, n) where * is zero or more batch dimensions
        consisting of invertible matrices.
    :return: The inverse of the input tensor.
    """
    return Tensor(inv(a.data))
