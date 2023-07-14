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

from typing import Optional, Tuple, Union

from nncf.common.tensor_new.tensor import Tensor
from nncf.common.tensor_new.tensor import tensor_func_dispatcher


def all(a: Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Tensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or tuple of axes along which to count non-zeros. Default is None,
       meaning that non-zeros will be counted along a flattened version of a.
    :return: A new boolean or tensor is returned unless out is specified,
      in which case a reference to out is returned.
    """
    return tensor_func_dispatcher("all", a, axis=axis)


def allclose(a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: True if the two arrays are equal within the given tolerance, otherwise False.
    """
    return tensor_func_dispatcher("allclose", a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def any(a: Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Tensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or tuple of axes along which to count non-zeros. Default is None,
       meaning that non-zeros will be counted along a flattened version of a.
    :return: A new boolean or tensor is returned unless out is specified,
      in which case a reference to out is returned.
    """
    return tensor_func_dispatcher("any", a, axis=axis)


def count_nonzero(a: Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Tensor:
    """
    Counts the number of non-zero values in the tensor input.

    :param a: The tensor for which to count non-zeros.
    :param axis: Axis or tuple of axes along which to count non-zeros. Default is None,
       meaning that non-zeros will be counted along a flattened version of a.
    :return: Number of non-zero values in the tensor along a given axis.
      Otherwise, the total number of non-zero values in the tensor is returned.
    """
    return tensor_func_dispatcher("count_nonzero", a, axis=axis)


def is_empty(a: Tensor) -> Tensor:
    """
    Return True if input tensor is empty.

    :param a: The input tensor.
    :return: True is tensor is empty, otherwise False.
    """
    return tensor_func_dispatcher("is_empty", a)


def isclose(a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: Returns a boolean tensor of where a and b are equal within the given tolerance.
    """
    return tensor_func_dispatcher("isclose", a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def maximum(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Element-wise maximum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    return tensor_func_dispatcher("maximum", x1, x2)


def minimum(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Element-wise minimum of tensor elements.

    :param input: The first input tensor.
    :param other: The second input tensor.
    :return: Output tensor.
    """
    return tensor_func_dispatcher("minimum", x1, x2)


def ones_like(a: Tensor) -> Tensor:
    """
    Return an tensor of ones with the same shape and type as a given tensor.

    :param a: The shape and data-type of a define these same attributes of the returned tensor.
    :return: Tensor of ones with the same shape and type as a.
    """
    return tensor_func_dispatcher("ones_like", a)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """
    Return elements chosen from x or y depending on condition.

    :param condition: Where True, yield x, otherwise yield y.
    :param x: Value at indices where condition is True.
    :param y: Value at indices where condition is False.
    :return: An tensor with elements from x where condition is True, and elements from y elsewhere.
    """
    return tensor_func_dispatcher("where", condition, x, y)


def zeros_like(a: Tensor) -> Tensor:
    """
    Return an tensor of zeros with the same shape and type as a given tensor.

    :param input: The shape and data-type of a define these same attributes of the returned tensor.
    :return: tensor of zeros with the same shape and type as a.
    """
    return tensor_func_dispatcher("zeros_like", a)
