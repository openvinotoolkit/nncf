# Copyright (c) 2023 Intel Corporation
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
from typing import Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import get_attribute_value

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import get_tensor_edge_name
from nncf.onnx.graph.onnx_graph import ONNXGraph
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


def _get_q_linear_params(node: onnx.NodeProto, onnx_graph: ONNXGraph) -> Tuple[np.dtype, int, np.ndarray, np.ndarray]:
    """


    :param node:
    :param onnx_graph:
    :return:
    """
    assert node.op_type in ONNXQuantizeLinearMetatype.get_all_aliases()
    scale = onnx_graph.get_tensor_value(node.input[1])
    zero_point = onnx_graph.get_tensor_value(node.input[2])
    dtype = zero_point.dtype
    axis = None
    for attr in node.attribute:
        if attr.name == "axis":
            axis = get_attribute_value(attr)
    return dtype, axis, scale, zero_point


def remove_node(node, onnx_graph):
    node_children = onnx_graph.get_children(node)
    for node_child in node_children:
        for input_id, input_obj in enumerate(node_child.input):
            if input_obj == node.output[0]:
                node_child.input[input_id] = node.input[0]
    onnx_graph.onnx_model.graph.node.remove(node)


def remove_unused_nodes(onnx_graph: ONNXGraph):
    model_outputs = [o.name for o in onnx_graph.get_model_outputs()]
    for node in onnx_graph.get_all_nodes():
        if not onnx_graph.get_nodes_by_input(node.output[0]) and node.output[0] not in model_outputs:
            onnx_graph.onnx_model.graph.node.remove(node)


def create_initializer_tensor(
    name: str, tensor_array: np.ndarray, data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name, data_type=data_type, dims=tensor_array.shape, vals=tensor_array.flatten().tolist()
    )
    return initializer_tensor


def compress_quantize_weights_transformation(model: onnx.ModelProto) -> onnx.ModelProto:
    onnx_graph = ONNXGraph(model)
    for node in onnx_graph.get_all_nodes():
        if node.op_type in ONNXQuantizeLinearMetatype.get_all_aliases():
            if get_tensor_edge_name(onnx_graph, node=node, port_id=0):
                init_tensor_name = None
                if not onnx_graph.has_tensor(node.input[0]):
                    e = onnx.utils.Extractor(model)
                    extracted = e.extract_model([], [node.input[0]])
                    from nncf.onnx.engine import ONNXEngine

                    engine = ONNXEngine(extracted)
                    outputs = engine.infer({})
                    original_weight = outputs[node.input[0]]
                else:
                    init_tensor_name = node.input[0]  # get_tensor_edge_name(onnx_graph, node=node, port_id=0)
                    original_weight = onnx_graph.get_tensor_value(init_tensor_name)

                dtype, axis, scale, zero_point = get_q_linear_params(node, onnx_graph)
                int8_weight = quantize_tensor(original_weight, dtype, axis, scale, zero_point)
                if init_tensor_name:
                    int8_weight_tensor = numpy_helper.from_array(int8_weight, name=init_tensor_name)

                    initializer = onnx_graph.get_tensor(init_tensor_name)
                    initializer.CopyFrom(int8_weight_tensor)

                    remove_node(node, onnx_graph)
                else:
                    init_tensor_name = get_tensor_edge_name(onnx_graph, node=node, port_id=0)
                    int8_weight_tensor = numpy_helper.from_array(int8_weight, name=init_tensor_name)

                    initializer = onnx_graph.get_tensor(init_tensor_name)
                    initializer.CopyFrom(int8_weight_tensor)
                    # initializzer = create_initializer_tensor(
                    #     'weight_tensor_' + str(cnt), int8_weight, np_dtype_to_tensor_dtype(int8_weight.dtype)
                    # )
                    # onnx_graph.onnx_model.graph.initializer.append(initializzer)
                    node.input[0] = init_tensor_name  # 'weight_tensor_' + str(cnt)

                    remove_node(node, onnx_graph)
    remove_unused_nodes(onnx_graph)
    return model
