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
from typing import List

import openvino.runtime as ov

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.openvino.graph.metatypes.groups import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVReadValueMetatype
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


def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
    """
    Get a list of NNCFNodes to use as start nodes for activation path tracing.

    :param nncf_graph: NNCFGraph to work with.
    :return: Target NNCFGraph input nodes.
    """
    return nncf_graph.get_input_nodes() + nncf_graph.get_nodes_by_metatypes([OVReadValueMetatype])


def remove_friendly_name_duplicates(model: ov.Model) -> ov.Model:
    """
    Removes diplicates of node names (friendly_name attribute) in the model.

    :param model: ov.Model instance to update.
    :return: Updated ov.Model without duplicated names.
    """
    rt_info_path = ["nncf", "friendly_names_were_updated"]
    friendly_names_flag = "True"
    if model.has_rt_info(rt_info_path) and model.get_rt_info(rt_info_path).value == friendly_names_flag:
        return model

    existing_names = set()
    for op in model.get_ops():
        friendly_name = op.get_friendly_name()
        if friendly_name in existing_names:
            friendly_name = friendly_name + "0"
            op.set_friendly_name(friendly_name)
        existing_names.add(friendly_name)
    model.set_rt_info(friendly_names_flag, rt_info_path)
    return model


def model_has_state(model: ov.Model) -> bool:
    """
    Returns True if model has state else False

    :param model: OpenVINO model
    :return: True if model has state else False
    """
    return len(model.get_sinks()) > 0


def copy_rt_info(model_source: ov.Model, model_dest: ov.Model, path: List[str]) -> None:
    """
    Checks and copies the rt_info from the source to destination model.

    :param model_source: ov.Model instance to copy rt_info from.
    :param model_dest: ov.Model instance to copy rt_info to.
    :param path: Path to rt_info.
    """
    if model_source.has_rt_info(path):
        source_rt_info = model_source.get_rt_info(path)
        model_dest.set_rt_info(source_rt_info, path)
