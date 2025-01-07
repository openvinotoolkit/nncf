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

from typing import Optional

from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns


def mean_per_channel(x: Tensor, axis: int, dtype: Optional[TensorDataType] = None) -> Tensor:
    """
    Computes the mean of elements across given channel dimension of Tensor.

    :param x: Tensor to reduce.
    :param axis: The channel dimensions to reduce.
    :param dtype: Type to use in computing the mean.
    :return: Reduced Tensor.
    """
    if len(x.shape) < 3:
        return fns.mean(x, axis=0, dtype=dtype)

    pos_axis = axis + x.ndim if axis < 0 else axis
    if pos_axis < 0 or pos_axis >= x.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {x.ndim}")
    axis = tuple(i for i in range(x.ndim) if i != pos_axis)
    return fns.mean(x, axis=axis, dtype=dtype)
