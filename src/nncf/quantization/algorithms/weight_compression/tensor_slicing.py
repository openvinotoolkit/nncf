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

from typing import Union

from nncf.tensor import Tensor

# slice is a built-in type, so we don't need to import it.
# slice_obj can be: an int (index), a slice (start:end), or a Tensor/Array (mask/indices)


def get_weight_slice(
    weight: Tensor,
    slice_obj: Union[int, slice, Tensor],
    is_transposed: bool,
) -> Tensor:
    """
    Generic helper to get a subset of weights along the input channel dimension.

    :param weight: The weight tensor.
    :param slice_obj: An integer index, a slice(start, end), or a boolean mask/index tensor.
    :param is_transposed: True if weight is [Out, In], False if [In, Out].
    :return: A slice of the weight tensor.
    """
    if is_transposed:
        return weight[:, slice_obj]
    return weight[slice_obj, :]


def set_weight_slice(
    weight: Tensor,
    slice_obj: Union[int, slice, Tensor],
    value: Union[Tensor, float, int],
    is_transposed: bool,
) -> None:
    """
    Generic helper to set a subset of weights along the input channel dimension.

    :param weight: The target tensor to modify in-place.
    :param slice_obj: An integer index, a slice(start, end), or a boolean mask/index tensor.
    :param value: The value(s) to assign.
    :param is_transposed: True if weight is [Out, In], False if [In, Out].
    """
    if is_transposed:
        weight[:, slice_obj] = value
    else:
        weight[slice_obj, :] = value
