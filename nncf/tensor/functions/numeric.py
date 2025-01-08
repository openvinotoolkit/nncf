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

import functools
from collections import deque
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from nncf.tensor import Tensor
from nncf.tensor import unwrap_tensor_data
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions.dispatcher import dispatch_list
from nncf.tensor.functions.dispatcher import get_numeric_backend_fn
from nncf.tensor.functions.dispatcher import tensor_guard
from nncf.tensor.tensor import TTensor


@functools.singledispatch
@tensor_guard
def device(a: Tensor) -> TensorDeviceType:
    """
    Return the device of the tensor.

    :param a: The input tensor.
    :return: The device of the tensor.
    """
    return device(a.data)


@functools.singledispatch
@tensor_guard
def backend(a: Tensor) -> TensorBackend:
    """
    Return the backend of the tensor.

    :param a: The input tensor.
    :return: The backend of the tensor.
    """
    return backend(a.data)


@functools.singledispatch
@tensor_guard
def squeeze(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Remove axes of length one from a.

    :param a: The input tensor.
    :param axis: Selects a subset of the entries of length one in the shape.
    :return: The input array, but with all or a subset of the dimensions of length 1 removed.
      This is always a itself or a view into a. Note that if all axes are squeezed,
      the result is a 0d array and not a scalar.
    """
    return Tensor(squeeze(a.data, axis=axis))


@functools.singledispatch
@tensor_guard
def flatten(a: Tensor) -> Tensor:
    """
    Return a copy of the tensor collapsed into one dimension.

    :param a: The input tensor.
    :return: A copy of the input tensor, flattened to one dimension.
    """
    return Tensor(flatten(a.data))


@functools.singledispatch
@tensor_guard
def max(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Return the maximum of an array or maximum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :param keepdim: If this is set to True, the axes which are reduced are left in the result as dimensions with size
        one. With this option, the result will broadcast correctly against the input array. False, by default.
    :return: Maximum of a.
    """
    return Tensor(max(a.data, axis, keepdims))


@functools.singledispatch
@tensor_guard
def min(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Return the minimum of an array or minimum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :param keepdim: If this is set to True, the axes which are reduced are left in the result as dimensions with size
        one. With this option, the result will broadcast correctly against the input array. False, by default.
    :return: Minimum of a.
    """
    return Tensor(min(a.data, axis, keepdims))


@functools.singledispatch
@tensor_guard
def abs(a: Tensor) -> Tensor:
    """
    Calculate the absolute value element-wise.

    :param a: The input tensor.
    :return: A tensor containing the absolute value of each element in x.
    """
    return Tensor(abs(a.data))


@functools.singledispatch
@tensor_guard
def astype(a: Tensor, data_type: TensorDataType) -> Tensor:
    """
    Copy of the tensor, cast to a specified type.

    :param a: The input tensor.
    :param dtype: Type code or data type to which the tensor is cast.

    :return: Copy of the tensor in specified type.
    """
    return Tensor(astype(a.data, data_type))


@functools.singledispatch
@tensor_guard
def dtype(a: Tensor) -> TensorDataType:
    """
    Return data type of the tensor.

    :param a: The input tensor.
    :return: The data type of the tensor.
    """
    return dtype(a.data)


@functools.singledispatch
@tensor_guard
def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Gives a new shape to a tensor without changing its data.

    :param a: Tensor to be reshaped.
    :param shape: The new shape should be compatible with the original shape.
    :return: Reshaped tensor.
    """
    return Tensor(reshape(a.data, shape))


@functools.singledispatch
@tensor_guard
def all(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical AND reduction is performed.
    :return: A new tensor.
    """
    return Tensor(all(a.data, axis=axis))


@functools.singledispatch
@tensor_guard
def allclose(
    a: Tensor, b: Union[Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
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
    return allclose(a.data, unwrap_tensor_data(b), rtol=rtol, atol=atol, equal_nan=equal_nan)


@functools.singledispatch
@tensor_guard
def any(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Test whether any tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical OR reduction is performed.
    :return: A new tensor.
    """
    return Tensor(any(a.data, axis))


@functools.singledispatch
@tensor_guard
def count_nonzero(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Counts the number of non-zero values in the tensor input.

    :param a: The tensor for which to count non-zeros.
    :param axis: Axis or tuple of axes along which to count non-zeros.
    :return: Number of non-zero values in the tensor along a given axis.
      Otherwise, the total number of non-zero values in the tensor is returned.
    """
    return Tensor(count_nonzero(a.data, axis))


@functools.singledispatch
@tensor_guard
def isempty(a: Tensor) -> bool:
    """
    Return True if input tensor is empty.

    :param a: The input tensor.
    :return: True if tensor is empty, otherwise False.
    """
    return isempty(a.data)


@functools.singledispatch
@tensor_guard
def isclose(
    a: Tensor, b: Union[Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> Tensor:
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
    return Tensor(
        isclose(
            a.data,
            unwrap_tensor_data(b),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    )


@functools.singledispatch
@tensor_guard
def maximum(x1: Tensor, x2: Union[Tensor, float]) -> Tensor:
    """
    Element-wise maximum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    return Tensor(maximum(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@tensor_guard
def minimum(x1: Tensor, x2: Union[Tensor, float]) -> Tensor:
    """
    Element-wise minimum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    return Tensor(minimum(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@tensor_guard
def ones_like(a: Tensor) -> Tensor:
    """
    Return a tensor of ones with the same shape and type as a given tensor.

    :param a: The shape and data-type of a define these same attributes of the returned tensor.
    :return: Tensor of ones with the same shape and type as a.
    """
    return Tensor(ones_like(a.data))


@functools.singledispatch
@tensor_guard
def where(condition: Tensor, x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
    """
    Return elements chosen from x or y depending on condition.

    :param condition: Where True, yield x, otherwise yield y.
    :param x: Value at indices where condition is True.
    :param y: Value at indices where condition is False.
    :return: A tensor with elements from x where condition is True, and elements from y elsewhere.
    """
    return Tensor(
        where(
            condition.data,
            unwrap_tensor_data(x),
            unwrap_tensor_data(y),
        )
    )


@functools.singledispatch
@tensor_guard
def zeros_like(a: Tensor) -> Tensor:
    """
    Return an tensor of zeros with the same shape and type as a given tensor.

    :param input: The shape and data-type of a define these same attributes of the returned tensor.
    :return: tensor of zeros with the same shape and type as a.
    """
    return Tensor(zeros_like(a.data))


@functools.singledispatch
def stack(x: List[Tensor], axis: int = 0) -> Tensor:
    """
    Stacks a list of Tensors rank-R tensors into one Tensor rank-(R+1) tensor.

    :param x: List of Tensors.
    :param axis: The axis to stack along.
    :return: Stacked Tensor.
    """
    if isinstance(x, (list, deque)):
        return Tensor(dispatch_list(stack, x, axis=axis))
    raise NotImplementedError(f"Function `stack` is not implemented for {type(x)}")


@functools.singledispatch
def concatenate(x: List[Tensor], axis: int = 0) -> Tensor:
    """
    Join a sequence of arrays along an existing axis.

    :param x: The arrays must have the same shape, except in the dimension corresponding to axis.
    :param axis: The axis along which the arrays will be joined. Default is 0.
    :return: The concatenated array.
    """
    if isinstance(x, (list, deque)):
        return Tensor(dispatch_list(concatenate, x, axis=axis))
    raise NotImplementedError(f"Function `concatenate` is not implemented for {type(x)}")


@functools.singledispatch
@tensor_guard
def unstack(a: Tensor, axis: int = 0) -> List[Tensor]:
    """
    Unstack a Tensor into list.

    :param a: Tensor to unstack.
    :param axis: The axis to unstack along.
    :return: List of Tensor.
    """
    res = unstack(a.data, axis=axis)
    return [Tensor(i) for i in res]


@functools.singledispatch
@tensor_guard
def moveaxis(a: Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> Tensor:
    """
    Move axes of an array to new positions.

    :param a: The array whose axes should be reordered.
    :param source: Original positions of the axes to move. These must be unique.
    :param destination: Destination positions for each of the original axes. These must also be unique.
    :return: Array with moved axes.
    """
    return Tensor(moveaxis(a.data, source, destination))


@functools.singledispatch
@tensor_guard
def mean(
    a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, dtype: TensorDataType = None
) -> Tensor:
    """
    Compute the arithmetic mean along the specified axis.

    :param a: Array containing numbers whose mean is desired.
    :param axis: Axis or axes along which the means are computed.
    :param keepdims: Destination positions for each of the original axes. These must also be unique.
    :param dtype: Type to use in computing the mean.
    :return: Array with moved axes.
    """
    return Tensor(mean(a.data, axis, keepdims, dtype))


@functools.singledispatch
@tensor_guard
def median(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the arithmetic median along the specified axis.

    :param a: Array containing numbers whose median is desired.
    :param axis: Axis or axes along which the medians are computed.
    :param keepdims: Destination positions for each of the original axes. These must also be unique.
    :return: Array with moved axes.
    """
    return Tensor(median(a.data, axis, keepdims))


@functools.singledispatch
@tensor_guard
def round(a: Tensor, decimals=0) -> Tensor:
    """
    Evenly round to the given number of decimals.

    :param a: Input data.
    :param decimals: Number of decimal places to round to (default: 0). If decimals is negative,
      it specifies the number of positions to the left of the decimal point.
    :return: An array of the same type as a, containing the rounded values.
    """
    return Tensor(round(a.data, decimals))


@functools.singledispatch
@tensor_guard
def power(a: Tensor, exponent: Union[Tensor, float]) -> Tensor:
    """
    Takes the power of each element in input with exponent and returns a tensor with the result.
    Exponent can be either a single float number or a broadcastable Tensor. In case exponent is
    a brodcastable tensor, the exponent is being broadcasted and the return tensor contains
    the power of each element in input with exponent elementwise.

    :param a: Input data.
    :param exponent: Exponent value.
    :return: The result of the power of each element in input with given exponent.
    """
    return Tensor(power(a.data, unwrap_tensor_data(exponent)))


@functools.singledispatch
@tensor_guard
def quantile(
    a: Tensor,
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """
    Compute the quantile(s) of the data along the specified axis.

    :param a: Given tensor.
    :params q: Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    :param axis: Axis or axes along which the quantiles are computed.
    :param keepdims: If True, the axes which are reduced are left in the result
        as dimensions with size one.
    :return: An tensor with quantiles, the first axis of the result corresponds
        to the quantiles, other axes of the result correspond to the quantiles values.
    """
    return Tensor(quantile(a.data, q, axis, keepdims))


@functools.singledispatch
@tensor_guard
def percentile(
    a: Tensor,
    q: Union[float, List[float]],
    axis: Union[int, Tuple[int, ...], List[int]],
    keepdims: bool = False,
) -> Tensor:
    """
    Compute the percentile(s) of the data along the specified axis.

    :param a: Given tensor.
    :params q: percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    :param axis: Axis or axes along which the percentiles are computed.
    :param keepdims: If True, the axes which are reduced are left in the result
        as dimensions with size one.
    :returns: The percentile(s) of the tensor elements.
    """
    return Tensor(percentile(a.data, q, axis, keepdims))


@functools.singledispatch
@tensor_guard
def _binary_op_nowarn(a: Tensor, b: Union[Tensor, float], operator_fn: Callable) -> Tensor:
    """
    Applies a binary operation with disable warnings.

    :param a: The first tensor.
    :param b: The second tensor.
    :param operator_fn: The binary operation function.
    :return: The result of the binary operation.
    """
    return Tensor(_binary_op_nowarn(a.data, unwrap_tensor_data(b), operator_fn))


@functools.singledispatch
@tensor_guard
def _binary_reverse_op_nowarn(a: Tensor, b: Union[Tensor, float], operator_fn: Callable) -> Tensor:
    """
    Applies a binary reverse operation with disable warnings.

    :param a: The first tensor.
    :param b: The second tensor.
    :param operator_fn: The binary operation function.
    :return: The result of the binary operation.
    """
    return Tensor(_binary_reverse_op_nowarn(a.data, unwrap_tensor_data(b), operator_fn))


@functools.singledispatch
@tensor_guard
def finfo(a: Tensor) -> TypeInfo:
    """
    Returns machine limits for tensor type.

    :param a: Tensor.
    :return: TypeInfo.
    """
    return finfo(a.data)


@functools.singledispatch
@tensor_guard
def clip(a: Tensor, a_min: Union[Tensor, float], a_max: Union[Tensor, float]) -> Tensor:
    """
    Clips all elements in input into the range [ a_min, a_max ]

    :param a: Tensor.
    :param a_min: A lower-bound of the range to be clamped to.
    :param a_max: An upper-bound of the range to be clamped to.
    :return: A clipped tensor with the elements of a, but where values < a_min are replaced with a_min,
        and those > a_max with a_max.
    """
    return Tensor(clip(a.data, unwrap_tensor_data(a_min), unwrap_tensor_data(a_max)))


@functools.singledispatch
@tensor_guard
def as_tensor_like(a: Tensor, data: Any) -> Tensor:
    """
    Converts the data into a tensor with the same data representation and hosted on the same device
    as the given tensor.

    :param a: A tensor for defining the data representation and the host device of the output tensor.
    :param data: Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
    :return: A tensor with the same data representation and hosted on the same device as a,
        and which has been initialized with data.
    """
    return Tensor(as_tensor_like(a.data, data))


@functools.singledispatch
@tensor_guard
def item(a: Tensor) -> Union[int, float, bool]:
    """
    Returns the value of this tensor as a standard Python number. This only works for tensors with one element.

    :param a: Tensor.
    :return: The value of this tensor as a standard Python number
    """
    if isinstance(a.data, (int, float, bool)):
        return a.data
    return item(a.data)


@functools.singledispatch
@tensor_guard
def sum(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Sum of tensor elements over a given axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements
        of the input tensor.
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
    :return: Returns the sum of all elements in the input tensor in the given axis.
    """
    return Tensor(sum(a.data, axis, keepdims))


@functools.singledispatch
@tensor_guard
def multiply(x1: Tensor, x2: Union[Tensor, float]) -> Tensor:
    """
    Multiply arguments element-wise.

    :param x1: The first input tensor.
    :param x2: The second input tensor or number.
    :return: The product of x1 and x2, element-wise.
    """
    return Tensor(multiply(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@tensor_guard
def var(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 0) -> Tensor:
    """
    Compute the variance along the specified axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which the variance is computed. The default is to compute the variance
        of the flattened tensor.
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
    :param ddof: “Delta Degrees of Freedom”: difference between the sample size and sample degrees of freedom.
        By default ddof is zero.
    :return: A new tensor containing the variance.
    """
    return Tensor(var(a.data, axis, keepdims, ddof))


@functools.singledispatch
@tensor_guard
def size(a: Tensor) -> int:
    """
    Return number of elements in the tensor.

    :param a: The input tensor
    :return: The size of the input tensor.
    """
    return size(a.data)


@functools.singledispatch
@tensor_guard
def matmul(x1: Tensor, x2: Union[Tensor, float]) -> Tensor:
    """
    Matrix multiplication.

    :param x1: The first input tensor.
    :param x2: The second input tensor or number.
    :return: The product of x1 and x2, matmul.
    """
    return Tensor(matmul(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@tensor_guard
def unsqueeze(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Add axes of length one to a.

    :param a: The input tensor.
    :param axis: Selects a subset of the entries of length one in the shape.
    :return: The input array, but with expanded shape with len 1 defined in axis.
    """
    return Tensor(unsqueeze(a.data, axis=axis))


@functools.singledispatch
@tensor_guard
def transpose(a: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """
    Returns an array with axes transposed.

    :param a: The input tensor.
    :param axes: List of permutations or None.
    :return: A tensor with permuted axes.
    """
    return Tensor(transpose(a.data, axes=axes))


@functools.singledispatch
@tensor_guard
def argsort(a: Tensor, axis: int = -1, descending: bool = False, stable: bool = False) -> Tensor:
    """
    Returns the indices that would sort an array.

    :param a: The input tensor.
    :param axis: Axis along which to sort. The default is -1 (the last axis).
    :param descending: Controls the sorting order (ascending or descending).
    :param stable: If True then the sorting routine becomes stable, preserving the order of equivalent elements.
        If False, the relative order of values which compare equal is not guaranteed. True is slower.
    :return: A tensor of indices that sort a along the specified axis.
    """
    return Tensor(argsort(a.data, axis=axis, descending=descending, stable=stable))


@functools.singledispatch
@tensor_guard
def diag(a: Tensor, k: int = 0) -> Tensor:
    """
    Returns the indices that would sort an array.

    :param a: The input tensor.
    :param k: Diagonal in question. The default is 0. Use k > 0 for diagonals above the main diagonal, and k < 0
        for diagonals below the main diagonal.
    :return: A tensor with the extracted diagonal.
    """
    return Tensor(diag(a.data, k=k))


@functools.singledispatch
@tensor_guard
def logical_or(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Computes the element-wise logical OR of the given input tensors.
    Zeros are treated as False and nonzeros are treated as True.

    :param x1: The input tensor.
    :param x2: The tensor to compute or with.
    :return: Result of elementwise or operation between input_ and other tensor.
    """
    return Tensor(logical_or(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@tensor_guard
def masked_mean(x: Tensor, mask: Tensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> Tensor:
    """
    Computes the masked mean of elements across given dimensions of Tensor.

    :param x: Tensor to reduce.
    :param mask: Boolean tensor that have the same shape as x. If an element in mask is True -
        it is skipped during the aggregation.
    :param axis: The dimensions to reduce.
    :param keepdims: If True, the axes which are reduced are left in the result
        as dimensions with size one.
    :return: Reduced Tensor.
    """
    return Tensor(masked_mean(x.data, mask.data, axis, keepdims))


@functools.singledispatch
@tensor_guard
def masked_median(x: Tensor, mask: Tensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> Tensor:
    """
    Computes the masked median of elements across given dimensions of Tensor.

    :param x: Tensor to reduce.
    :param axis: The dimensions to reduce.
    :param mask: Boolean tensor that have the same shape as x. If an element in mask is True -
        it is skipped during the aggregation.
    :param keepdims: If True, the axes which are reduced are left in the result
        as dimensions with size one.
    :return: Reduced Tensor.
    """
    return Tensor(masked_median(x.data, mask.data, axis, keepdims))


@functools.singledispatch
@tensor_guard
def expand_dims(a: Tensor, axis: Union[int, Tuple[int, ...], List[int]]) -> Tensor:
    """
    Expand the shape of an array.
    Insert a new axis that will appear at the axis position in the expanded array shape.

    :param a: Input array.
    :param axis: Position in the expanded axes where the new axis (or axes) is placed.

    :return: View of a with the number of dimensions increased.
    """
    return Tensor(expand_dims(a.data, axis))


@functools.singledispatch
@tensor_guard
def clone(a: Tensor) -> Tensor:
    """
    Return a copy of the tensor.

    :param a: The input tensor.
    :return: The copied tensor.
    """
    return Tensor(clone(a.data))


@functools.singledispatch
@tensor_guard
def searchsorted(a: Tensor, v: Tensor, side: str = "left", sorter: Optional[Tensor] = None) -> Tensor:
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted tensor a such that, if the corresponding elements in v were inserted
    before the indices, the order of a would be preserved.

    :param a: 1-D tensor. If sorter is None, then it must be sorted in ascending order,
        otherwise sorter must be an array of indices that sort it.
    :param v: Values to insert into a.
    :param side: If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index. Defaults to 'left'.
    :param sorter: Optional array of integer indices that sort array a into ascending order, defaults to None.
    :return: Tensor of insertion points with the same shape as v.
    """
    return Tensor(searchsorted(a.data, v.data, side, unwrap_tensor_data(sorter)))


def zeros(
    shape: Tuple[int, ...],
    *,
    backend: TensorBackend,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> Tensor:
    """
    Return a new array of given shape and type, filled with zeros.

    :param shape: Shape of the new array
    :param backend: The backend type for which the zero tensor is required.
    :param dtype: The data type of the returned tensor, If dtype is not given,
        then the default data type is determined by backend.
    :param device: The device on which the tensor will be allocated, If device is not given,
        then the default device is determined by backend.
    :return: A tensor filled with zeros of the specified shape and data type.
    """
    return Tensor(get_numeric_backend_fn("zeros", backend)(shape, dtype=dtype, device=device))


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    backend: TensorBackend,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> Tensor:
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    :param n: Number of rows in the output.
    :param m: Number of columns in the output. If None, defaults to n.
    :param backend: The backend type for which the eye tensor is required.
    :param dtype: The data type of the returned tensor, If dtype is not given,
        then the default data type is determined by backend.
    :param device: The device on which the tensor will be allocated, If device is not given,
        then the default device is determined by backend.
    :return: A tensor where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.
    """
    return Tensor(get_numeric_backend_fn("eye", backend)(n, m, dtype=dtype, device=device))


def arange(
    start: float,
    end: Optional[float] = None,
    step: float = 1,
    *,
    backend: TensorBackend,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> Tensor:
    """
    Returns a tensor with a sequence of numbers in the specified range.

    This function generates a tensor containing a sequence of numbers starting from
    `start` to `end` with a step size of `step`, using the specified backend.

    :param start: The starting value of the sequence.
    :param end: The end value of the sequence (exclusive). If None, the sequence will start at 0 and end at `start`.
    :param step: The step size between each value in the sequence, defaults to 1.
    :param backend: The backend type for which the tensor is required.
    :param dtype: The data type of the returned tensor If dtype is not given, infer the data type
        from the other input arguments.
    :param device: The device on which the tensor will be allocated, If device is not given,
        then the default device is determined by backend.
    :return: A tensor containing the sequence of numbers.
    """
    args = (0, start, step) if end is None else (start, end, step)
    return Tensor(get_numeric_backend_fn("arange", backend)(*args, dtype=dtype, device=device))


def from_numpy(ndarray: np.ndarray, *, backend: TensorBackend) -> Tensor:
    """
    Creates a Tensor from a numpy.ndarray, sharing the memory with the given NumPy array.

    :param ndarray: The numpy.ndarray to share memory with.
    :param backend: The backend type for which the tensor is required.
    :return: A Tensor object that shares memory with the NumPy array.
    """
    if backend == TensorBackend.numpy:
        return Tensor(ndarray)
    return Tensor(get_numeric_backend_fn("from_numpy", backend)(ndarray))


@functools.singledispatch
@tensor_guard
def log2(a: Tensor) -> Tensor:
    """
    Base-2 logarithm of a.

    :param a: The input tensor.
    :return: A tensor containing the base-2 logarithm of each element in a.
    """
    return Tensor(log2(a.data))


@functools.singledispatch
@tensor_guard
def ceil(a: Tensor) -> Tensor:
    """
    Return the ceiling of the input, element-wise.

    :param a: Input data.
    :return: An array of the same type as a, containing the ceiling values.
    """
    return Tensor(ceil(a.data))


def tensor(
    data: Union[TTensor, Sequence[float]],
    *,
    backend: TensorBackend,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> Tensor:
    """
    Creates a tensor from the given data.

    :param data: The data for the tensor.
    :param backend: The backend type for which the tensor is required.
    :param dtype: The data type of the returned tensor, If dtype is not given,
        then the default data type is determined by backend.
    :param device: The device on which the tensor will be allocated, If device is not given,
        then the default device is determined by backend.
    :return: A tensor created from the given data.
    """
    return Tensor(get_numeric_backend_fn("tensor", backend)(data, dtype=dtype, device=device))
