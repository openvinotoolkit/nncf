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

from typing import Optional, TypeVar

from nncf.common.tensor_new.tensor import Tensor
from nncf.common.tensor_new.tensor import tensor_func_dispatcher

TensorType = TypeVar("TensorType")


def maximum(target: Tensor, other: Tensor) -> Tensor:
    return tensor_func_dispatcher("maximum", target, other)


def minimum(target: Tensor, other: Tensor) -> Tensor:
    return tensor_func_dispatcher("minimum", target, other)


def zeros_like(target: Tensor) -> Tensor:
    return tensor_func_dispatcher("zeros_like", target)


def ones_like(target: Tensor) -> Tensor:
    return tensor_func_dispatcher("ones_like", target)


def count_nonzero(target: Tensor, axis: Optional[TensorType] = None) -> Tensor:
    return tensor_func_dispatcher("count_nonzero", target, axis=axis)


def all(target: Tensor) -> Tensor:
    return tensor_func_dispatcher("all", target)


def any(target: Tensor) -> Tensor:
    return tensor_func_dispatcher("any", target)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return tensor_func_dispatcher("where", condition, x, y)


def is_empty(target: Tensor) -> Tensor:
    return tensor_func_dispatcher("is_empty", target)
