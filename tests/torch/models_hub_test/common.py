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

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import pytest
import torch
from _pytest.mark import ParameterSet
from torch import nn

import nncf
from nncf.common.graph import NNCFGraph
from nncf.torch.model_creation import wrap_model

ExampleType = Union[Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]]


@pytest.mark.models_hub
class BaseTestModel(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> Tuple[nn.Module, ExampleType]:
        """
        Load model by name and create example of input.

        :param model_name: Name of model.
        :return: A Tuple
          - Model instance
          - Example of input
        """

    @staticmethod
    def check_graph(graph: NNCFGraph) -> None:
        """
        Check that the graph contains only one connected component.

        :param graph: The graph.
        """
        nx_graph = graph._get_graph_for_visualization()
        nx_graph = nx_graph.to_undirected()
        num_connected_components = len(list(nx.connected_components(nx_graph)))
        assert num_connected_components == 1, f"Disconnected graph, {num_connected_components} connected components"

    def nncf_wrap(self, model_name) -> None:
        """
        Wrap model and check the graph.

        :param model_name: Model name.
        """
        torch.manual_seed(0)

        fw_model, example = self.load_model(model_name)

        example_input = None
        if isinstance(example, (list, tuple)):
            example_input = tuple([torch.tensor(x) for x in example])
        elif isinstance(example, dict):
            example_input = {k: torch.tensor(v) for k, v in example.items()}
        assert example_input is not None

        nncf_model = wrap_model(fw_model, example_input, trace_parameters=True)

        self.check_graph(nncf_model.nncf.get_original_graph())


@dataclass
class ModelInfo:
    model_name: Optional[str]
    model_link: Optional[str]
    mark: Optional[str]
    reason: Optional[str]


def idfn(val):
    if isinstance(val, ModelInfo):
        return val.model_name
    return None


def get_models_list(path: Path) -> List[ModelInfo]:
    """
    Get list of test model from file.

    :param path: Path to file.
    :return: List of models.
    """
    models = []
    with path.open() as f:
        for model_info in f:
            model_info = model_info.rstrip()
            # skip comment in model scope file
            if model_info.startswith("#"):
                continue
            mark = None
            reason = None
            model_link = None

            splitted = model_info.split(",")
            if len(splitted) == 1:
                model_name = splitted[0]
            elif len(splitted) == 2:
                model_name, model_link = splitted
            elif len(splitted) == 4:
                model_name, model_link, mark, reason = splitted
                if model_link == "none":
                    model_link = None
                assert mark in ["skip", "xfail"], "Incorrect failure mark for model info {}".format(model_info)
            else:
                raise nncf.ValidationError(
                    f"Incorrect model info `{model_info}`. It must contain either 1, 2 or 3 fields."
                )
            models.append(ModelInfo(model_name, model_link, mark, reason))

    return models


def get_model_params(path: Path) -> List[Union[ModelInfo, ParameterSet]]:
    """
    Get test cases from the file.

    :param path: Path to file.
    :return: Test parameters.
    """
    model_list = get_models_list(path)
    params = []
    for mi in model_list:
        if mi.mark == "skip":
            params.append(pytest.param(mi, marks=pytest.mark.skip(reason=mi.reason)))
        elif mi.mark == "xfail":
            params.append(pytest.param(mi, marks=pytest.mark.xfail(reason=mi.reason)))
        else:
            params.append(mi)
    return params
