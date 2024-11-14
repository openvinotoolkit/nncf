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

from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.functions import numeric

from ..definitions import TensorBackend
from ..definitions import TensorDeviceType
from .numpy_numeric import DTYPE_MAP as DTYPE_MAP_NP
from .numpy_numeric import DTYPE_MAP_REV as DTYPE_MAP_REV_NP

DTYPE_MAP = {
    TensorDataType.float16: ov.Type.f16,
    TensorDataType.bfloat16: ov.Type.bf16,
    TensorDataType.float32: ov.Type.f32,
    TensorDataType.float64: ov.Type.f64,
    TensorDataType.int8: ov.Type.i8,
    TensorDataType.int32: ov.Type.i32,
    TensorDataType.int64: ov.Type.i64,
    TensorDataType.uint8: ov.Type.u8,
    TensorDataType.uint4: ov.Type.u4,
    TensorDataType.int4: ov.Type.i4,
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


@numeric.device.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorDeviceType:
    return TensorDeviceType.CPU


@numeric.backend.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorBackend:
    return TensorBackend.ov


@numeric.astype.register(ov.Tensor)
def _(a: ov.Tensor, dtype: TensorDataType) -> ov.Tensor:
    if a.get_element_type() in [ov.Type.bf16, ov.Type.i4, ov.Type.u4] or dtype in [
        TensorDataType.bfloat16,
        TensorDataType.int4,
        TensorDataType.uint4,
    ]:
        return _astype_ov(a, dtype)
    return ov.Tensor(a.data.astype(DTYPE_MAP_NP[dtype]))


@numeric.dtype.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.get_element_type()]


@numeric.size.register(ov.Tensor)
def _(a: ov.Tensor) -> int:
    return a.size


@numeric.reshape.register(ov.Tensor)
def _(a: ov.Tensor, shape: Union[int, Tuple[int, ...]]) -> ov.Tensor:
    return ov.Tensor(a.data.reshape(shape), shape, a.get_element_type())


@numeric.to_backend.register(np.ndarray)
def _(a: np.ndarray, b: TensorBackend) -> Union[np.ndarray, ov.Tensor]:
    if b == TensorBackend.numpy:
        return a
    if b != TensorBackend.ov:
        raise ValueError("Not supported backend")
    return ov.Tensor(a, a.shape, DTYPE_MAP[DTYPE_MAP_REV_NP[a.dtype]])


@numeric.to_backend.register(ov.Tensor)
def _(a: ov.Tensor, b: TensorBackend) -> Union[np.ndarray, ov.Tensor]:
    if b == TensorBackend.ov:
        return a
    if b != TensorBackend.numpy:
        raise ValueError("Not supported backend")

    # Cannot convert bfloat16, uint4, int4 to numpy directly
    a_dtype = DTYPE_MAP_REV[a.get_element_type()]
    if a_dtype in [TensorDataType.bfloat16, TensorDataType.uint4, TensorDataType.int4]:
        dtype = TensorDataType.float32
        if a_dtype == TensorDataType.uint4:
            dtype = TensorDataType.uint8
        elif a_dtype == TensorDataType.int4:
            dtype = TensorDataType.int8
        a = _astype_ov(a, dtype)

    return a.data


def _astype_ov(a: ov.Tensor, dtype: TensorDataType) -> ov.Tensor:
    from nncf.quantization.algorithms.weight_compression.openvino_modeling import OVModelParameters
    from nncf.quantization.algorithms.weight_compression.openvino_modeling import get_astype_model

    a_dtype = DTYPE_MAP_REV[a.get_element_type()]

    model = get_astype_model(
        OVModelParameters(
            input_dtypes={"input": a_dtype},
            output_dtypes={"output": dtype},
            dynamic_shapes=False,
            recompile=True,
            release_memory=True,
            share_inputs=True,
            share_outputs=True,
            return_ov_tensors=True,
        ),
        tuple(a.shape),
    )
    return model([Tensor(a)])[0].data
