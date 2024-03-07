# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import tests.post_training.test_templates.helpers as helpers
from nncf.torch import wrap_model
from nncf.torch.extractor import extract_fused_subgraph_for_node


@pytest.mark.parametrize(
    "model_cls, node_name",
    (
        (helpers.ConvBiasBNTestModel, "ConvBiasBNTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.ConvBNTestModel, "ConvBNTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.ConvTestModel, "ConvTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.CustomConvBNTestModel, "CustomConvBNTestModel/CustomConv[conv]/conv2d_0"),
        (helpers.CustomConvTestModel, "CustomConvTestModel/CustomConv[conv]/conv2d_0"),
    ),
)
def test_extract_fused_subgraph_for_node(model_cls, node_name):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model = wrap_model(model_cls().eval(), example_input=example_input, trace_parameters=True)
    graph = model.nncf.get_graph()
    node = graph.get_node_by_name(node_name)
    extracted_module = extract_fused_subgraph_for_node(node, model)

    with torch.no_grad():
        model = model.eval()
        extracted_module = extracted_module.eval()
        ret1 = model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.any(torch.isclose(ret1, ret2))
