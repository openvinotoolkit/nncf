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

from typing import TypeVar

import nncf.common.tensor_ops as tensor_ops
from nncf.common.tensor_new import Tensor

TensorType = TypeVar("TensorType")


def maximum(target: Tensor, other: TensorType) -> Tensor:
    return Tensor(tensor_ops.maximum(Tensor.safe_get_tensor_data(target.data), Tensor.safe_get_tensor_data(other)))


def minimum(target: Tensor, other: TensorType) -> Tensor:
    return Tensor(tensor_ops.minimum(Tensor.safe_get_tensor_data(target.data), Tensor.safe_get_tensor_data(other)))


def zeros_like(target: Tensor) -> Tensor:
    return Tensor(tensor_ops.zeros_like(Tensor.safe_get_tensor_data(target)))


def ones_like(target: Tensor) -> Tensor:
    return Tensor(tensor_ops.ones_like(Tensor.safe_get_tensor_data(target)))


def all(target: Tensor) -> Tensor:
    return Tensor(tensor_ops.all(Tensor.safe_get_tensor_data(target)))


def any(target: TensorType) -> Tensor:
    return Tensor(tensor_ops.any(Tensor.safe_get_tensor_data(target)))


def where(condition: TensorType, x: TensorType, y: TensorType) -> TensorType:
    return Tensor(
        tensor_ops.where(
            Tensor.safe_get_tensor_data(condition),
            Tensor.safe_get_tensor_data(x),
            Tensor.safe_get_tensor_data(y),
        )
    )


def is_empty(target: TensorType) -> TensorType:
    return Tensor(tensor_ops.is_empty(Tensor.safe_get_tensor_data(target)))
