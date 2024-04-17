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

from nncf.torch import wrap_model
from nncf.torch.model_analyzer import get_fused_bias_value
from nncf.torch.model_transformer import update_fused_bias
from tests.post_training.test_templates.helpers import ConvBiasBNTestModel
from tests.post_training.test_templates.helpers import ConvBNTestModel
from tests.post_training.test_templates.helpers import ConvTestModel


@pytest.mark.parametrize(
    "model_cls, ref",
    (
        (ConvTestModel, [0.1000, 1.0000]),  # conv.bias
        (ConvBNTestModel, [0.1000, 1.0000]),  # bn.bias
        (ConvBiasBNTestModel, [0.1600, 3.6000]),  # conv.bias*bn.weight + bn.bias
    ),
)
def test_get_fused_bias_value(model_cls, ref):
    model = wrap_model(model_cls(), torch.ones(model_cls.INPUT_SIZE))

    graph = model.nncf.get_graph()
    target_node = graph.get_nodes_by_types("conv2d")[0]

    bias = get_fused_bias_value(target_node, model)
    assert torch.all(torch.isclose(bias, torch.tensor(ref)))


@pytest.mark.parametrize(
    "model_cls",
    (
        (ConvTestModel),  # conv.bias
        (ConvBNTestModel),  # bn.bias
        (ConvBiasBNTestModel),  # conv.bias*bn.weight + bn.bias
    ),
)
def test_update_fused_bias(model_cls):
    model = wrap_model(model_cls(), torch.ones(model_cls.INPUT_SIZE))
    ref_new_bias = torch.tensor([-1.0, -1.0])
    graph = model.nncf.get_graph()
    target_node = graph.get_nodes_by_types("conv2d")[0]

    update_fused_bias(target_node.node_name, ref_new_bias, model)
    bias = get_fused_bias_value(target_node, model)
    assert torch.all(torch.isclose(bias, ref_new_bias))

    if model_cls == ConvTestModel:
        assert torch.all(torch.isclose(model.conv.bias, ref_new_bias))
    if model_cls == ConvBNTestModel:
        assert model.conv.bias is None
        assert torch.all(torch.isclose(model.bn.bias, ref_new_bias))
    if model_cls == ConvBiasBNTestModel:
        assert torch.all(torch.isclose(model.conv.bias, torch.tensor([0.3000, 1.3000])))
        assert torch.all(torch.isclose(model.bn.bias, torch.tensor([-1.0600, -3.6000])))
        assert torch.all(torch.isclose(model.conv.bias * model.bn.weight + model.bn.bias, ref_new_bias))
