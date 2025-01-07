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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch


class NodeType(Enum):
    const = "const"
    fn_call = "function_call"
    input = "input"
    output = "output"

    def __str__(self) -> str:
        return self.value


class TensorSource(Enum):
    buffer = "buffer"
    function = "function"
    input = "input"
    output = "output"
    parameter = "parameter"

    def __str__(self) -> str:
        return self.value


@dataclass
class TensorMeta:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> TensorMeta:
        return TensorMeta(tensor.dtype, tuple(tensor.shape), tensor.requires_grad)


@dataclass
class ConstMeta:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    name_in_model: str

    @staticmethod
    def from_tensor(tensor: torch.Tensor, name_in_model: str) -> ConstMeta:
        return ConstMeta(tensor.dtype, tuple(tensor.shape), name_in_model)


@dataclass
class InOutMeta:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    name: str

    @staticmethod
    def from_tensor(tensor: torch.Tensor, name: str) -> InOutMeta:
        return InOutMeta(tensor.dtype, tuple(tensor.shape), name)


@dataclass
class FunctionMeta:
    op_name: str
    fn_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


@dataclass
class EdgeMeta:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    input_port: int
    output_port: int

    @staticmethod
    def from_tensor(tensor: torch.Tensor, input_port: int, output_port: int) -> EdgeMeta:
        return EdgeMeta(tensor.dtype, tuple(tensor.shape), input_port, output_port)


@dataclass
class TensorInfo:
    tensor_source: TensorSource
    shape: Tuple[int, ...]
    dtype: torch.dtype
    output_port_id: int
    source_node_id: Optional[int]
    name_in_model: Optional[str]
