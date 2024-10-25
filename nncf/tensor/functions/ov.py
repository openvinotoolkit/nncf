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
from typing import Tuple, Union

import numpy as np
import openvino as ov

from nncf.tensor import TensorDataType
from nncf.tensor.functions import numeric

from ..definitions import TensorBackend
from .numpy_numeric import DTYPE_MAP as NP_DTYPE_MAP

DTYPE_MAP = {
    TensorDataType.float16: ov.Type.f16,
    TensorDataType.bfloat16: ov.Type.bf16,
    TensorDataType.float32: ov.Type.f32,
    TensorDataType.float64: ov.Type.f64,
    TensorDataType.int8: ov.Type.i8,
    TensorDataType.int32: ov.Type.i32,
    TensorDataType.int64: ov.Type.i64,
    TensorDataType.uint8: ov.Type.u8,
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


def _bf16_to_fp32(a: ov.Tensor) -> ov.Tensor:
    assert a.get_element_type() == ov.Type.bf16 and a.data.dtype == np.float16

    a = a.data.view(np.uint16)

    res = a.astype(np.uint32)
    res = (
        ((res & 0x8000) << 16)  # Move sign bit to bit 31
        | ((res & 0x7F80) << 16)  # Move exponent to bits 30-23
        | ((res & 0x007F) << 16)
    )  # Move fraction to bits 22-0
    res = res.view(np.float32)

    res = ov.Tensor(res)
    return res


@numeric.backend.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorBackend:
    return TensorBackend.ov


@numeric.astype.register(ov.Tensor)
def _(a: ov.Tensor, dtype: TensorDataType) -> ov.Tensor:
    if dtype == TensorDataType.bfloat16:
        raise ValueError("Not supported conversion")
    if a.get_element_type() == ov.Type.bf16:
        a = _bf16_to_fp32(a)
    return ov.Tensor(a.data.astype(NP_DTYPE_MAP[dtype]))


@numeric.dtype.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.get_element_type()]


@numeric.size.register(ov.Tensor)
def _(a: ov.Tensor) -> int:
    return a.size


@numeric.reshape.register(ov.Tensor)
def _(a: ov.Tensor, shape: Union[int, Tuple[int, ...]]) -> ov.Tensor:
    return ov.Tensor(a.data.reshape(shape), shape, a.get_element_type())


@numeric.to_backend.register(ov.Tensor)
def _(a: ov.Tensor, b: TensorBackend) -> np.ndarray:
    if b != TensorBackend.numpy:
        raise ValueError("Not supported backend")
    return a.data
