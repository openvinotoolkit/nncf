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
import copy
from collections import deque
from typing import Dict

import numpy as np
import onnx
from onnx.external_data_helper import _get_initializer_tensors
from onnx.external_data_helper import set_external_data

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint


def remove_fq_from_inputs(model: onnx.ModelProto, nncf_graph: NNCFGraph) -> onnx.ModelProto:
    """
    This method removes the activation Quantizer nodes from the model.
    It's needed for the further bias shift calculation that relates on quantized weights.

    :param model: onnx.ModelProto instance.
    :param nncf_graph: NNCFGraph instance.
    :return: onnx.ModelProto instance without activation Quantizer nodes.
    """
    transformation_layout = TransformationLayout()
    model_transformer = ModelTransformerFactory.create(model)

    seen_nodes = []
    nodes_queue = deque(nncf_graph.get_input_nodes())
    while nodes_queue:
        current_node = nodes_queue.popleft()
        current_node_name = current_node.node_name

        if current_node_name in seen_nodes:
            continue

        seen_nodes.append(current_node_name)
        if current_node.metatype in [ONNXQuantizeLinearMetatype, ONNXDequantizeLinearMetatype]:
            target_point = ONNXTargetPoint(TargetType.LAYER, current_node_name, 0)
            command = ONNXQDQNodeRemovingCommand(target_point)
            transformation_layout.register(command)
        nodes_queue.extend(nncf_graph.get_next_nodes(current_node))

    return model_transformer.transform(transformation_layout)


def extract_raw_data_from_model(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """
    Extracts raw data from the initializer tensors of an ONNX model.

    This method iterates through all the initializer tensor protos in the given ONNX model,
    converting the content of the `raw_data` field into NumPy array. It then clears the `raw_data`
    field in each tensor.

    :param model: The ONNX model to extract raw data from.
    :return: A dictionary with tensor names as keys and NumPy arrays of raw data as values.
    """
    tensors = _get_initializer_tensors(model)
    data = {}
    for tensor in tensors:
        if not tensor.HasField("raw_data"):
            continue
        # TODO(andrey-churkin): Handle the case when the model is loaded without data
        # `onnx.load("model.onnx", load_external_data=False)`.

        # TODO(andrey-churkin): Probably, we should convert the NumPy array into an `ort.OrtValue`
        # here as follows: `OrtValue.ortvalue_from_numpy(numpy_tensor)`.
        data[tensor.name] = onnx.numpy_helper.to_array(tensor)

        # We should call the `set_external_data()`` method here; otherwise, we will get an error during
        # session creation because we can't replace a non-external initializer with external data.
        set_external_data(tensor, location="foo.bin")
        tensor.ClearField("raw_data")

    return data


def insert_raw_data_into_model(model: onnx.ModelProto, data: Dict[str, np.ndarray]) -> None:
    """
    Updates the `raw_data` field of the initializer tensors in the given ONNX model using the
    NumPy arrays from the provided dictionary.

    :param model: The ONNX model to insert raw data into.
    :param data: A dictionary where the keys are the names of the initializer tensors and the values
        are NumPy arrays to be assigned to the `raw_data` field of the corresponding tensors.
    """
    tensors = _get_initializer_tensors(model)
    for tensor in tensors:
        numpy_array = data.get(tensor.name, None)
        if numpy_array is not None:
            # TODO(andrey-churkin): Should we preserve the external data options here?
            tensor_proto = onnx.numpy_helper.from_array(numpy_array, tensor.name)
            tensor.CopyFrom(tensor_proto)


class ONNXModel:
    def __init__(self, model: onnx.ModelProto, data: Dict[str, np.ndarray]):
        """
        :param model: The ONNX model without raw data loaded into its initializer tensors.
        :param data: A dictionary where the keys are the names of the initializer tensors and
            the values are NumPy arrays representing the `raw_data` field for each corresponding tensor.
        """
        self._model = model
        self._data = data

    @property
    def model_proto(self) -> onnx.ModelProto:
        return self._model

    @property
    def tensors(self) -> Dict[str, np.ndarray]:
        return self._data

    @classmethod
    def from_model(cls, model: onnx.ModelProto) -> "ONNXModel":
        """
        Creates an instance of ONNXModel from a given ONNX model.

        :param model: The ONNX model.
        :return: An ONNXModel instance containing the model and the extracted raw data from the
            initializer tensors.
        """
        # The `extract_raw_data_from_model()` method modifies the model passed to it,
        # so we should create a copy of the original model here.
        copy_model = copy.deepcopy(model)
        data = extract_raw_data_from_model(copy_model)

        # TODO(andrey-churkin): Complete all preprocessing here
        copy_model = onnx.shape_inference.infer_shapes(copy_model)

        return cls(copy_model, data)

    def export(self) -> onnx.ModelProto:
        """
        Exports the ONNXModel instance to an ONNX model, inserting the raw data into
        its initializer tensors.

        :return: The ONNX model with the raw data inserted into its initializer tensors.
        """
        copy_model = copy.deepcopy(self._model)
        insert_raw_data_into_model(copy_model, self._data)
        return copy_model
