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
from typing import Any, Dict, List, Tuple

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.quantization.algorithms.post_training.algorithm import TModel


class PostTrainingBackend(ABC):
    @property
    @abstractmethod
    def if_node_metatype(self):
        """_summary_"""

    @staticmethod
    @abstractmethod
    def get_if_node_input_names(model: TModel, if_node: NNCFNode, subgraph_port_id: int) -> str:
        """ """

    @staticmethod
    @abstractmethod
    def create_update_subgraph_command(target_point, subgraph_model):
        """

        :param target_point:
        :param subgraph_model:
        :return:
        """

    @staticmethod
    @abstractmethod
    def create_extract_if_subgraph_command(if_node, port_id):
        """ """

    @staticmethod
    @abstractmethod
    def create_output_insertion_commands(model, if_node):
        """"""

    @staticmethod
    @abstractmethod
    def dump_model(model: TModel, dir: Path, backend_params: Dict[str, Any]) -> None:
        """
        Save a model to a directory. Backend params are used to determine the model name to dump.

        :param model: Model to dump.
        :param dir: Directory path.
        :param backend_params: Backend specific parameters.
        """
