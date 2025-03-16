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
from typing import Any, Dict, Tuple, Union

import onnx
from numpy.typing import NDArray

from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric

ONNX_DTYPE_MAP: Dict[TensorDataType, int] = {
    TensorDataType.float16: onnx.TensorProto.DataType.FLOAT16,
    TensorDataType.bfloat16: onnx.TensorProto.DataType.BFLOAT16,
    TensorDataType.float32: onnx.TensorProto.DataType.FLOAT,
    TensorDataType.float64: onnx.TensorProto.DataType.DOUBLE,
    TensorDataType.int8: onnx.TensorProto.DataType.INT8,
    TensorDataType.int32: onnx.TensorProto.DataType.INT32,
    TensorDataType.int64: onnx.TensorProto.DataType.INT64,
    TensorDataType.uint8: onnx.TensorProto.DataType.UINT8,
}

DTYPE_MAP_REV = {v: k for k, v in ONNX_DTYPE_MAP.items()}