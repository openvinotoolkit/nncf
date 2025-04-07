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

from collections import deque

import onnx

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.onnx_helper import get_children
from nncf.onnx.graph.onnx_helper import get_children_node_mapping
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
