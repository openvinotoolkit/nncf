"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import NamedTuple, Union
from copy import deepcopy
import onnx
from onnx import TensorProto  # pylint:disable=no-name-in-module
from onnx.external_data_helper import uses_external_data
from nncf.onnx.graph.onnx_graph import ONNXGraph
import numpy as np
import tempfile

from pathlib import Path

# pylint: disable=no-member

# TODO(kshpv): This maaping should be removed after upgrading to the onnx > 1.12

TensorDtypeMap = NamedTuple(
    "TensorDtypeMap", [("np_dtype", np.dtype), ("storage_dtype", int), ("name", str)]
)

# tensor_dtype: (numpy type, storage type, string name)
TENSOR_TYPE_MAP = {
    int(TensorProto.FLOAT): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.FLOAT), "TensorProto.FLOAT"
    ),
    int(TensorProto.UINT8): TensorDtypeMap(
        np.dtype("uint8"), int(TensorProto.INT32), "TensorProto.UINT8"
    ),
    int(TensorProto.INT8): TensorDtypeMap(
        np.dtype("int8"), int(TensorProto.INT32), "TensorProto.INT8"
    ),
    int(TensorProto.UINT16): TensorDtypeMap(
        np.dtype("uint16"), int(TensorProto.INT32), "TensorProto.UINT16"
    ),
    int(TensorProto.INT16): TensorDtypeMap(
        np.dtype("int16"), int(TensorProto.INT32), "TensorProto.INT16"
    ),
    int(TensorProto.INT32): TensorDtypeMap(
        np.dtype("int32"), int(TensorProto.INT32), "TensorProto.INT32"
    ),
    int(TensorProto.INT64): TensorDtypeMap(
        np.dtype("int64"), int(TensorProto.INT64), "TensorProto.INT64"
    ),
    int(TensorProto.BOOL): TensorDtypeMap(
        np.dtype("bool"), int(TensorProto.INT32), "TensorProto.BOOL"
    ),
    int(TensorProto.FLOAT16): TensorDtypeMap(
        np.dtype("float16"), int(TensorProto.UINT16), "TensorProto.FLOAT16"
    ),
    # Native numpy does not support bfloat16 so now use float32 for bf16 values
    # TODO ONNX should dirtectly use bfloat16 for bf16 values after numpy has supported bfloat16 type
    int(TensorProto.BFLOAT16): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT16), "TensorProto.BFLOAT16"
    ),
    int(TensorProto.DOUBLE): TensorDtypeMap(
        np.dtype("float64"), int(TensorProto.DOUBLE), "TensorProto.DOUBLE"
    ),
    int(TensorProto.COMPLEX64): TensorDtypeMap(
        np.dtype("complex64"), int(TensorProto.FLOAT), "TensorProto.COMPLEX64"
    ),
    int(TensorProto.COMPLEX128): TensorDtypeMap(
        np.dtype("complex128"), int(TensorProto.DOUBLE), "TensorProto.COMPLEX128"
    ),
    int(TensorProto.UINT32): TensorDtypeMap(
        np.dtype("uint32"), int(TensorProto.UINT32), "TensorProto.UINT32"
    ),
    int(TensorProto.UINT64): TensorDtypeMap(
        np.dtype("uint64"), int(TensorProto.UINT64), "TensorProto.UINT64"
    ),
    int(TensorProto.STRING): TensorDtypeMap(
        np.dtype("object"), int(TensorProto.STRING), "TensorProto.STRING"
    ),
}


def load_model_topology_with_zeros_weights(model_path: Union[str, Path]) -> onnx.ModelProto:
    """
    Loads onnx model and fills the all external tensors by zeros values.

    :param model_path: Path to the onnx model to load.
    :return: Onnx model with filled the all external tensors by random values.
    """
    model = onnx.load_model(model_path, load_external_data=False)
    onnx_graph = ONNXGraph(model)
    for tensor in onnx_graph.onnx_model.graph.initializer:
        if uses_external_data(tensor):
            np_dtype = TENSOR_TYPE_MAP[tensor.data_type].np_dtype
            np_tensor = np.zeros(list(tensor.dims)).astype(np_dtype)
            tensor_with_zeros = onnx.numpy_helper.from_array(np_tensor, name=tensor.name)
            tensor.CopyFrom(tensor_with_zeros)
            del tensor.external_data[:]
            tensor.data_location = TensorProto.DEFAULT
    return model


def save_model_without_tensors(model: onnx.ModelProto, model_path: Path) -> None:
    """
    Saves the onnx model topology to 'model_path'. Saved model does not contain tensors.

    :param model: Onnx model to save.
    :param model_path: Path to save the onnx model.
    :return: None.
    """
    tensors_location = Path('tensors')
    copy_model = deepcopy(model)
    with tempfile.TemporaryDirectory() as tmpfile:
        onnx.save_model(copy_model, model_path, save_as_external_data=True, location=Path(tmpfile) / tensors_location)
