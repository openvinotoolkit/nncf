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

import openvino.runtime as ov

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.openvino.graph.metatypes.groups import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator


def remove_fq_from_inputs(model: ov.Model, graph: NNCFGraph) -> ov.Model:
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
        if current_node.metatype in FAKE_QUANTIZE_OPERATIONS:
            command = OVCommandCreator.create_command_to_remove_quantizer(current_node)
            transformation_layout.register(command)
        nodes_queue.extend(graph.get_next_nodes(current_node))

    return model_transformer.transform(transformation_layout)
