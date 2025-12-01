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

from nncf.tensor import Tensor


def slice_weight(weight: Tensor, start: int, end: int, transpose_b: bool) -> Tensor:
    """
    Return a view/clone of the requested block without transposing the whole tensor.

    If transpose_b is True, weight layout is [out_features, in_features]
    and we return weight[:, start:end] (in_features slice).
    If transpose_b is False, layout is [in_features, out_features]
    and we return weight[start:end, :] (in_features slice).

    :param weight: The weight tensor to slice.
    :param start: Start index for the slice (inclusive).
    :param end: End index for the slice (exclusive).
    :param transpose_b: Whether the weight is transposed (True) or not (False).
    :return: A slice of the weight tensor.
    """
    if transpose_b:
        return weight[:, start:end]
    else:
        return weight[start:end, :]


def extract_weight_column(weight: Tensor, index: int, transpose_b: bool) -> Tensor:
    """
    Extract a single column/row from weight based on transpose_b.

    If transpose_b is True: returns weight[:, index] (a column)
    If transpose_b is False: returns weight[index, :] (a row)

    :param weight: The weight tensor to extract from.
    :param index: The index of the column/row to extract.
    :param transpose_b: Whether the weight is transposed (True) or not (False).
    :return: A single column or row from the weight tensor.
    """
    if transpose_b:
        return weight[:, index]
    else:
        return weight[index, :]


def assign_weight_slice(target_weight: Tensor, start: int, end: int, block: Tensor, transpose_b: bool) -> None:
    """
    Assign block back to target_weight in the same orientation used by slice_weight.
    This performs in-place assignment.

    :param target_weight: The target weight tensor to assign to.
    :param start: Start index for the slice (inclusive).
    :param end: End index for the slice (exclusive).
    :param block: The block of data to assign.
    :param transpose_b: Whether the weight is transposed (True) or not (False).
    """
    if transpose_b:
        target_weight[:, start:end] = block
    else:
        target_weight[start:end, :] = block


def assign_weight_column(target_weight: Tensor, index: int, column: Tensor, transpose_b: bool) -> None:
    """
    Assign a single column/row back to target_weight.
    This performs in-place assignment.

    :param target_weight: The target weight tensor to assign to.
    :param index: The index of the column/row to assign.
    :param column: The column/row data to assign.
    :param transpose_b: Whether the weight is transposed (True) or not (False).
    """
    if transpose_b:
        target_weight[:, index] = column
    else:
        target_weight[index, :] = column


def zero_mask_columns(weight: Tensor, mask: Tensor, transpose_b: bool) -> None:
    """
    Zero out columns/rows based on boolean mask.

    If transpose_b is True: zeros weight[:, mask] (columns)
    If transpose_b is False: zeros weight[mask, :] (rows)

    :param weight: The weight tensor to modify in-place.
    :param mask: Boolean mask indicating which columns/rows to zero.
    :param transpose_b: Whether the weight is transposed (True) or not (False).
    """
    if transpose_b:
        weight[:, mask] = 0
    else:
        weight[mask, :] = 0

