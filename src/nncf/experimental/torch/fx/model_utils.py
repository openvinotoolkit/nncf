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

import torch.fx

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.transformations import node_removal_transformation_builder
from nncf.torch.graph.operator_metatypes import QUANTIZE_NODE_TYPES
from nncf.torch.graph.transformations.commands import PTTargetPoint


def remove_fq_from_inputs(model: torch.fx.GraphModule, graph: NNCFGraph) -> torch.fx.GraphModule:
    """
    This method removes the activation Fake Quantize nodes from the model.
    It's needed for the further bias shift calculation that relates on quantized weights.

    :param model: ov.Model instance.
    :param graph: NNCFGraph instance.
    :return: ov.Model instance without activation Fake Quantize nodes.
    """
    transformation_layout = TransformationLayout()
    model_transformer = ModelTransformerFactory.create(model)

    seen_nodes = []
    nodes_queue = deque(graph.get_input_nodes())
    while nodes_queue:
        current_node = nodes_queue.popleft()
        current_node_name = current_node.node_name

        if current_node_name in seen_nodes:
            continue

        seen_nodes.append(current_node_name)
        if current_node.node_type in QUANTIZE_NODE_TYPES:
            transformation = node_removal_transformation_builder(current_node, input_port_id=0)
            transformation_layout.register(FXApplyTransformationCommand(transformation))
        nodes_queue.extend(graph.get_next_nodes(current_node))

    return model_transformer.transform(transformation_layout)


_TARGET_TYPE_TO_FX_INS_TYPE_MAP = {
    TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
    TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
}


def get_target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
    """
    Creates torch-specific target point.

    :param target_type: Target point target type.
    :param target_node_name: Target node name to use in the target point.
    :param port_id: Target port id.
    :return: Torch-specific target point.
    """
    if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
        port_id = None
    if target_type in _TARGET_TYPE_TO_FX_INS_TYPE_MAP:
        target_type = _TARGET_TYPE_TO_FX_INS_TYPE_MAP[target_type]
    return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)
