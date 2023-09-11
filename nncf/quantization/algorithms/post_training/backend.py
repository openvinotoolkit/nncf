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

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import Command
from nncf.quantization.algorithms.post_training.algorithm import TModel


class PostTrainingBackend(ABC):
    @property
    @abstractmethod
    def if_node_metatype(self):
        """
        Property for Metatype of If node.
        """

    @staticmethod
    @abstractmethod
    def get_if_subgraph_input_names(model: TModel, if_node: NNCFNode, if_submodel_condition: bool) -> List[str]:
        """
        Returns input names of subgraph of If node.

        :param model: Main Model.
        :param if_node: If node.
        :param if_submodel_condition: If node subgraph condition.
        :return: Input names of subgraph.
        """

    @staticmethod
    @abstractmethod
    def get_if_cond_input_name(model: TModel, if_node: NNCFNode) -> str:
        """
        Returns input names of condition of If node.

        :param model: Model.
        :param if_node: If node.
        :return: Name of edge.
        """

    @staticmethod
    @abstractmethod
    def create_update_subgraph_command(
        if_node: NNCFNode, if_submodel_condition: bool, subgraph_model: TModel
    ) -> Command:
        """
        Returns command for updating If node subgraph.

        :param if_node: If node.
        :param if_submodel_condition: If node subgraph condition.
        :param subgraph_model: Submodel to insert.
        :return: Command
        """

    @staticmethod
    @abstractmethod
    def create_extract_if_subgraph_command(if_node: NNCFNode, if_submodel_condition: bool) -> Command:
        """
        Returns extract if subgraph command.

        :param if_node: If node.
        :param if_submodel_condition: If node submodel condition.
        :return: Extract if subgraph command.
        """

    @staticmethod
    @abstractmethod
    def create_output_insertion_commands(model: TModel, if_node: NNCFNode) -> List[Command]:
        """
        Returns output insertion commands for If node.

        :param model: Main model.
        :param if_node: If node.
        :return: Output insertion commands.
        """

    @staticmethod
    @abstractmethod
    def dump_submodel(model: TModel, directory: Path, if_op: NNCFNode, if_submodel_condition: bool) -> None:
        """
        Save a submodel to a directory.

        :param model: Model to dump.
        :param directory: Directory path.
        :param if_op: If node.
        :param if_submodel_condition: If submodel condition.
        """
