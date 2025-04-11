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
