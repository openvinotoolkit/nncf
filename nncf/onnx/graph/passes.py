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

import onnx

from nncf.onnx.graph.onnx_helper import get_children
from nncf.onnx.graph.onnx_helper import get_children_node_mapping


def eliminate_nop_cast(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Inspects the provided ONNX model to identify and remove any 'No-op' (no-operation)
    cast nodes, which are operations that do not change the data type of their input.

    :param model: The ONNX model to be processed.
    :return: The ONNX model with the redundant cast nodes removed.
    """
    tensor_name_to_info = {
        tensor.name: tensor
        for tensor in (*model.graph.value_info, *model.graph.input, *model.graph.output, *model.graph.initializer)
    }
    redundant_cast_nodes = []
    for node in model.graph.node:
        if node.op_type == "Cast":
            to_attr = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_attr = onnx.helper.get_attribute_value(attr)

            if to_attr is None:
                continue

            inp = node.input[0]
            info = tensor_name_to_info[inp]
            if info.type.tensor_type.elem_type == to_attr:
                redundant_cast_nodes.append(node)

    value_infos = {i.name: i for i in model.graph.value_info}
    input_name_to_nodes_map = get_children_node_mapping(model)

    for cast_node in redundant_cast_nodes:
        # Unlink Cast node from the graph
        children = get_children(cast_node, input_name_to_nodes_map)
        for child in children:
            for i, input_name in enumerate(child.input):
                if input_name == cast_node.output[0]:
                    child.input[i] = cast_node.input[0]

        # Remove Cast node from the graph
        model.graph.value_info.remove(value_infos[cast_node.output[0]])
        model.graph.node.remove(cast_node)

    return model


def apply_preprocess_passes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Preprocesses the provided ONNX model for quantization.

    This method performs the following steps:
        1. Infers shapes in the model.
        2. Removes redundant 'No-op' cast nodes from the model.

    :param model: The ONNX model to be preprocessed.
    :return: A preprocessed ONNX model, ready for quantization.
    """
    preprocessed_model = onnx.shape_inference.infer_shapes(model)
    # The `eliminate_nop_cast` pass should be applied after onnx.shape_inference.infer_shapes() call.
    # Otherwise, not all no-op Cast nodes will be found.
    preprocessed_model = eliminate_nop_cast(preprocessed_model)
    return preprocessed_model
