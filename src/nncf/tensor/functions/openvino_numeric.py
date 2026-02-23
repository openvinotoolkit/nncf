# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import openvino as ov  # type: ignore
from numpy.typing import NDArray

from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP_REV as DTYPE_MAP_REV_NUMPY

DTYPE_MAP: dict[TensorDataType, ov.Type] = {
    TensorDataType.nf4: ov.Type.nf4,
    TensorDataType.f4e2m1: ov.Type.f4e2m1,
    TensorDataType.f8e8m0: ov.Type.f8e8m0,
    TensorDataType.f8e4m3: ov.Type.f8e4m3,
    TensorDataType.f8e5m2: ov.Type.f8e5m2,
    TensorDataType.float16: ov.Type.f16,
    TensorDataType.bfloat16: ov.Type.bf16,
    TensorDataType.float32: ov.Type.f32,
    TensorDataType.float64: ov.Type.f64,
    TensorDataType.int8: ov.Type.i8,
    TensorDataType.int32: ov.Type.i32,
    TensorDataType.int64: ov.Type.i64,
    TensorDataType.uint16: ov.Type.u16,
    TensorDataType.uint32: ov.Type.u32,
    TensorDataType.uint8: ov.Type.u8,
    TensorDataType.uint4: ov.Type.u4,
    TensorDataType.int4: ov.Type.i4,
}

NATIVE_OV_CAST_DTYPES = [
    TensorDataType.bfloat16,
    TensorDataType.int4,
    TensorDataType.uint4,
    TensorDataType.nf4,
    TensorDataType.f4e2m1,
    TensorDataType.f8e8m0,
    TensorDataType.f8e4m3,
    TensorDataType.f8e5m2,
]

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


def from_numpy(a: NDArray[Any]) -> ov.Tensor:
    """
    Convert a numpy array to an OpenVINO tensor.

    :param a: Numpy array to convert.
    :return: OpenVINO tensor.
    """
    return ov.Tensor(a, a.shape, DTYPE_MAP[DTYPE_MAP_REV_NUMPY[a.dtype]])


@numeric.device.register
def _(a: ov.Tensor) -> TensorDeviceType:
    return TensorDeviceType.CPU


@numeric.backend.register
def _(a: ov.Tensor) -> TensorBackend:
    return TensorBackend.ov


@numeric.astype.register
def _(a: ov.Tensor, dtype: TensorDataType) -> ov.Tensor:
    a_dtype = DTYPE_MAP_REV[a.get_element_type()]
    if a_dtype in NATIVE_OV_CAST_DTYPES or dtype in NATIVE_OV_CAST_DTYPES:
        # Cast using OpenVINO because the target or source dtype requires special handling
        return _astype_ov(a, dtype)
    return ov.Tensor(numeric.astype(a.data, dtype).data)


@numeric.dtype.register
def _(a: ov.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.get_element_type()]


@numeric.size.register
def _(a: ov.Tensor) -> int:
    return a.size


@numeric.reshape.register
def _(a: ov.Tensor, shape: int | tuple[int, ...]) -> ov.Tensor:
    return ov.Tensor(a.data.reshape(shape), shape, a.get_element_type())


@numeric.as_numpy_tensor.register
def _(a: ov.Tensor) -> NDArray[Any]:
    # Cannot convert bfloat16, uint4, int4, nf4, f4e2m1, f8e8m0, f8e4m3, f8e5m2 to numpy directly
    a_dtype = DTYPE_MAP_REV[a.get_element_type()]
    if a_dtype in NATIVE_OV_CAST_DTYPES:
        dtype = TensorDataType.float32
        if a_dtype == TensorDataType.uint4:
            dtype = TensorDataType.uint8
        elif a_dtype == TensorDataType.int4:
            dtype = TensorDataType.int8
        a = _astype_ov(a, dtype)
    return a.data


@numeric.finfo.register
def _(a: ov.Tensor) -> TypeInfo:
    return numeric.finfo(a.data)


def _astype_ov(a: ov.Tensor, dtype: TensorDataType) -> ov.Tensor:
    """
    Cast to a different data type using an OpenVINO model.

    :param a: Tensor to cast.
    :param dtype: Data type to cast to.
    :return: Casted openvino tensor.
    """
    from nncf.openvino.optimized_functions import astype

    return astype(Tensor(a), dtype).data
