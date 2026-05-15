# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import TypeVar

import pytest
import torch

from nncf.common.factory import build_graph
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from tests.torch.test_models.synthetic import ConcatSDPABlock

TModel = TypeVar("TModel")


class TemplateTestUnifiedScales:
    @property
    @abstractmethod
    def get_backend_specific_model(self, model: TModel) -> TModel:
        """
        Convert and return backend specific Model

        :param model: Model (for example in PT) to be converted to backend specific model
        :return: Backend specific Model
        """

    @pytest.mark.parametrize(
        "model_cls, unified_group, unified_group_nncf_network",
        ((ConcatSDPABlock, [["x", "y"]], [["/nncf_model_input_0", "/nncf_model_input_1"]]),),
    )
    def test_unified_groups(
        self, model_cls: TModel, unified_group: list[list[str]], unified_group_nncf_network: list[list[str]]
    ):
        backend_model = self.get_backend_specific_model(model_cls())
        if isinstance(backend_model, torch.nn.Module) and not isinstance(backend_model, torch.fx.GraphModule):
            unified_group = unified_group_nncf_network

        nncf_graph = build_graph(backend_model)
        algo = MinMaxQuantization()
        algo._set_backend_entity(backend_model)
        _, groups = algo._get_quantization_target_points(backend_model, nncf_graph)
        assert [[target.target_node_name for target in groups] for groups in groups] == unified_group
