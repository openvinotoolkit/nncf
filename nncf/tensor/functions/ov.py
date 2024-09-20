import numpy as np
import openvino as ov

from nncf.tensor import TensorDataType
from nncf.tensor.functions import numeric
from .numpy_numeric import DTYPE_MAP as NP_DTYPE_MAP
from ..definitions import TensorBackend

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


@numeric.backend.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorBackend:
    return TensorBackend.ov


@numeric.astype.register(ov.Tensor)
def _(a: ov.Tensor, dtype: TensorDataType) -> np.ndarray:
    return a.data.astype(NP_DTYPE_MAP[dtype])


@numeric.dtype.register(ov.Tensor)
def _(a: ov.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.get_element_type()]


@numeric.size.register(ov.Tensor)
def _(a: ov.Tensor) -> int:
    return a.size
