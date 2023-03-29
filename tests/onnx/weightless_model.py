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

from typing import Union
from copy import deepcopy
import onnx
from onnx import TensorProto  # pylint:disable=no-name-in-module
from onnx.external_data_helper import uses_external_data
from nncf.onnx.graph.onnx_graph import ONNXGraph
import numpy as np
import tempfile

from pathlib import Path

# pylint: disable=no-member


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
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
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
