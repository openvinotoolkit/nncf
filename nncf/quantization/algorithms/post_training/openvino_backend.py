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

from pathlib import Path
from typing import List

import openvino.runtime as ov

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.metatypes.openvino_metatypes import OVIfMetatype
from nncf.openvino.graph.transformations.commands import OVExtractIfSubgraphCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVUpdateIfSubgraphCommand
from nncf.quantization.algorithms.post_training.backend import PostTrainingBackend


class OVPostTrainingBackend(PostTrainingBackend):
    @staticmethod
    def _get_if_submodel_port_id(if_submodel_condition: bool):
        return int(not if_submodel_condition)

    @property
    def if_node_metatype(self):
        return OVIfMetatype

    @staticmethod
    def get_if_subgraph_input_names(model: ov.Model, if_node: NNCFNode, if_submodel_condition: bool) -> List[str]:
        input_names = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        input_indices = [
            desc.input_index
            for desc in ov_node.get_input_descriptions(
                OVPostTrainingBackend._get_if_submodel_port_id(if_submodel_condition)
            )
        ]
        input_names.extend([ov_node.input_values()[index].any_name for index in input_indices])
        return input_names

    @staticmethod
    def get_if_cond_input_name(model: ov.Model, if_node: NNCFNode) -> str:
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        return ov_node.input_values()[0].any_name

    @staticmethod
    def create_update_subgraph_command(
        if_node: NNCFNode, if_submodel_condition: bool, subgraph_model: ov.Model
    ) -> OVUpdateIfSubgraphCommand:
        target_point = OVTargetPoint(
            TargetType.LAYER, if_node.node_name, OVPostTrainingBackend._get_subgraph_port_id(if_submodel_condition)
        )
        return OVUpdateIfSubgraphCommand(target_point, subgraph_model)

    @staticmethod
    def create_extract_if_subgraph_command(
        if_node: NNCFNode, if_submodel_condition: bool
    ) -> OVExtractIfSubgraphCommand:
        return OVExtractIfSubgraphCommand(if_node.node_name, if_submodel_condition)

    @staticmethod
    def create_output_insertion_commands(model: ov.Model, if_node: NNCFNode) -> List[OVOutputInsertionCommand]:
        commands = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        for port_id in range(len(ov_node.inputs())):
            commands.append(
                OVOutputInsertionCommand(OVTargetPoint(TargetType.PRE_LAYER_OPERATION, if_node.node_name, port_id))
            )
        return commands

    @staticmethod
    def dump_submodel(model: ov.Model, directory: str, if_op: NNCFNode, if_submodel_condition: bool) -> None:
        name = if_op.node_name.replace("/", "")
        postfix = "then" if if_submodel_condition else "else"
        model_name = f"{name}_{postfix}.xml"
        model_path = Path(directory) / model_name
        ov.serialize(model, model_path)
