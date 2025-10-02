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
from onnx.reference.ops import load_op

from nncf.onnx.graph.onnx_helper import get_children
from nncf.onnx.graph.onnx_helper import get_children_node_mapping
from nncf.onnx.graph.onnx_helper import get_node_attr_value


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


def compress_quantize_weights_transformation(model: onnx.ModelProto):
    """
    Transforms the model by folding `QuantizeLinear` nodes with constant inputs
    into precomputed, quantized initializers.

    This transformation finds `QuantizeLinear` nodes with constant inputs
    (i.e., inputs present in the model's initializers), precomputes their quantized values,
    updates the initializer with these results, and removes the corresponding
    `QuantizeLinear` nodes from the graph.

    :param model: The model to be transformed.
    """
    initializer = {x.name: x for x in model.graph.initializer}
    nodes_to_remove = []

    version = max(model.opset_import[0].version, 19)
    QuantizeLinear = load_op("", "QuantizeLinear", version)

    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue

        x_name, y_scale_name = node.input[:2]
        # `y_zero_point` is an optional input for the `QuantizeLinear` operation.
        y_zero_point_name = node.input[2] if len(node.input) > 2 else None

        if x_name not in initializer:
            continue

        nodes_to_remove.append(node)

        # Quantize
        x = onnx.numpy_helper.to_array(initializer[x_name])
        y_scale = onnx.numpy_helper.to_array(initializer[y_scale_name])

        y_zero_point = None
        if y_zero_point_name:
            y_zero_point = onnx.numpy_helper.to_array(initializer[y_zero_point_name])

        axis = get_node_attr_value(node, "axis")
        if version < 21:
            # onnx.reference.ops.op_quantize_linear.QuantizeLinear_19
            y = QuantizeLinear.eval(x, y_scale, y_zero_point, axis=axis)
        else:
            # onnx.reference.ops.op_quantize_linear.QuantizeLinear_21
            block_size = get_node_attr_value(node, "block_size")
            y = QuantizeLinear.eval(x, y_scale, y_zero_point, axis=axis, block_size=block_size)

        # Update an existing initializer. The new name is the name of the `QuantizeLinear` output.
        tensor_proto = onnx.numpy_helper.from_array(y, name=node.output[0])
        initializer[x_name].CopyFrom(tensor_proto)

    # `QuantizeLinear` and `DequantizeLinear` nodes share initializers on ports 1 and 2,
    # so these initializers should not be removed.
    for x in nodes_to_remove:
        model.graph.node.remove(x)
