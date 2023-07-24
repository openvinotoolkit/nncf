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

import functools
from typing import Optional, Tuple, TypeVar, Union

import nncf.experimental.common.tensor as tensor  # pylint: disable=consider-using-from-import
from nncf.experimental.common.tensor.enums import TensorDataType
from nncf.experimental.common.tensor.enums import TensorDeviceType

TTensor = TypeVar("TTensor")
T = TypeVar("T")  # TODO: Verify


@functools.singledispatch
def device(a: TTensor) -> TensorDeviceType:
    """
    Return the device of the tensor.

    :param a: The input tensor.
    :return: The device of the tensor.
    """
    return device(a.data)


@functools.singledispatch
def squeeze(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:
    """
    Remove axes of length one from a.

    :param a: The input tensor.
    :param axis: Selects a subset of the entries of length one in the shape.
    :return: The input array, but with all or a subset of the dimensions of length 1 removed.
      This is always a itself or a view into a. Note that if all axes are squeezed,
      the result is a 0d array and not a scalar.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(squeeze(a.data, axis=axis))
    return NotImplemented(f"Function `squeeze` is not implemented for {type(a)}")


@functools.singledispatch
def flatten(a: TTensor) -> TTensor:
    """
    Return a copy of the tensor collapsed into one dimension.

    :param a: The input tensor.
    :return: A copy of the input tensor, flattened to one dimension.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(flatten(a.data))
    return NotImplemented(f"Function `flatten` is not implemented for {type(a)}")


@functools.singledispatch
def max(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Return the maximum of an array or maximum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :return: Maximum of a.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(max(a.data, axis))
    return NotImplemented(f"Function `max` is not implemented for {type(a)}")


@functools.singledispatch
def min(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Return the minimum of an array or minimum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :return: Minimum of a.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(min(a.data, axis))
    return NotImplemented(f"Function `min` is not implemented for {type(a)}")


@functools.singledispatch
def abs(a: TTensor) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Calculate the absolute value element-wise.

    :param a: The input tensor.
    :return: A tensor containing the absolute value of each element in x.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(abs(a.data))
    return NotImplemented(f"Function `abs` is not implemented for {type(a)}")


@functools.singledispatch
def astype(a: TTensor, data_type: TensorDataType) -> TTensor:
    """
    Copy of the tensor, cast to a specified type.

    :param a: The input tensor.
    :param dtype: Type code or data-type to which the tensor is cast.

    :return: Copy of the tensor in specified type.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(astype(a.data, data_type))
    return NotImplemented(f"Function `dtype` is not implemented for {type(a)}")


@functools.singledispatch
def dtype(a: TTensor) -> TensorDataType:
    """
    Return data type fo the tensor.

    :param a: The input tensor.
    :return: The data type fo the tensor.
    """
    if isinstance(a, tensor.Tensor):
        return dtype(a.data)
    return NotImplemented(f"Function `dtype` is not implemented for {type(a)}")


@functools.singledispatch
def reshape(a: TTensor, shape: T) -> TTensor:
    """
    :param a:
    :param shape:
    :return:
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(reshape(a.data, shape))
    return NotImplemented(f"Function `dtype` is not implemented for {type(a)}")


###############################################################################


@functools.singledispatch
def all(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical AND reduction is performed.
    :return: A new boolean or tensor is returned unless out is specified,
      in which case a reference to out is returned.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(all(a.data, axis=axis))
    return NotImplemented(f"Function `any` is not implemented for {type(a)}")


@functools.singledispatch
def allclose(a: TTensor, b: TTensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> TTensor:
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
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(
            allclose(
                a.data,
                tensor.unwrap_tensor_data(b),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )
    return NotImplemented(f"Function `allclose` is not implemented for {type(a)}")


@functools.singledispatch
def any(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical OR reduction is performed.
    :return: A new boolean or tensor is returned unless out is specified,
      in which case a reference to out is returned.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(any(a.data, axis))
    return NotImplemented(f"Function `any` is not implemented for {type(a)}")


@functools.singledispatch
def count_nonzero(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:
    """
    Counts the number of non-zero values in the tensor input.

    :param a: The tensor for which to count non-zeros.
    :param axis: Axis or tuple of axes along which to count non-zeros.
    :return: Number of non-zero values in the tensor along a given axis.
      Otherwise, the total number of non-zero values in the tensor is returned.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(count_nonzero(a.data, axis))
    return NotImplemented(f"Function `count_nonzero` is not implemented for {type(a)}")


@functools.singledispatch
def isempty(a: TTensor) -> bool:
    """
    Return True if input tensor is empty.

    :param a: The input tensor.
    :return: True is tensor is empty, otherwise False.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(isempty(a.data))
    return NotImplemented(f"Function `isempty` is not implemented for {type(a)}")


@functools.singledispatch
def isclose(a: TTensor, b: TTensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> TTensor:
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
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(
            isclose(
                a.data,
                tensor.unwrap_tensor_data(b),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )
    return NotImplemented(f"Function `where` is not implemented for {type(a)}")


@functools.singledispatch
def maximum(x1: TTensor, x2: TTensor) -> TTensor:
    """
    Element-wise maximum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    if isinstance(x1, tensor.Tensor):
        return tensor.Tensor(
            maximum(
                x1.data,
                tensor.unwrap_tensor_data(x2),
            )
        )
    return NotImplemented(f"Function `maximum` is not implemented for {type(x1)}")


@functools.singledispatch
def minimum(x1: TTensor, x2: TTensor) -> TTensor:
    """
    Element-wise minimum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    if isinstance(x1, tensor.Tensor):
        return tensor.Tensor(
            minimum(
                x1.data,
                tensor.unwrap_tensor_data(x2),
            )
        )
    return NotImplemented(f"Function `minimum` is not implemented for {type(x1)}")


@functools.singledispatch
def ones_like(a: TTensor) -> TTensor:
    """
    Return an tensor of ones with the same shape and type as a given tensor.

    :param a: The shape and data-type of a define these same attributes of the returned tensor.
    :return: Tensor of ones with the same shape and type as a.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(ones_like(a.data))
    return NotImplemented(f"Function `ones_like` is not implemented for {type(a)}")


@functools.singledispatch
def where(condition: TTensor, x: TTensor, y: TTensor) -> TTensor:
    """
    Return elements chosen from x or y depending on condition.

    :param condition: Where True, yield x, otherwise yield y.
    :param x: Value at indices where condition is True.
    :param y: Value at indices where condition is False.
    :return: An tensor with elements from x where condition is True, and elements from y elsewhere.
    """
    if isinstance(condition, tensor.Tensor):
        return tensor.Tensor(
            where(
                condition.data,
                tensor.unwrap_tensor_data(x),
                tensor.unwrap_tensor_data(y),
            )
        )
    return NotImplemented(f"Function `where` is not implemented for {type(condition)}")


@functools.singledispatch
def zeros_like(a: TTensor) -> TTensor:
    """
    Return an tensor of zeros with the same shape and type as a given tensor.

    :param input: The shape and data-type of a define these same attributes of the returned tensor.
    :return: tensor of zeros with the same shape and type as a.
    """
    if isinstance(a, tensor.Tensor):
        return tensor.Tensor(zeros_like(a.data))
    return NotImplemented(f"Function `zeros_like` is not implemented for {type(a)}")
