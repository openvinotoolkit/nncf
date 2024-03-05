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

from dataclasses import dataclass
from typing import Dict, Tuple

import pytest
import torch
from torch import nn

import tests.post_training.test_templates.helpers as helpers
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.torch import wrap_model
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_data_on_port
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import get_potential_fused_node
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.model_graph_manager import is_node_with_fused_bias
from nncf.torch.model_graph_manager import set_const_data
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.helpers import create_conv


@dataclass
class ModelDesc:
    model: NNCFNetwork
    graph: NNCFGraph
    node: NNCFNode
    node_name: str
    model_name: str

    def __init__(self, model_cls: type, node_name: str):
        self.model = wrap_model(model_cls(), example_input=torch.ones(model_cls.INPUT_SIZE), trace_parameters=True)
        self.graph = self.model.nncf.get_graph()
        self.node = self.graph.get_node_by_name(node_name)
        self.node_name = node_name


MODELS_LIST = [
    "ConvBiasBNTestModel",
    "ConvBNTestModel",
    "ConvTestModel",
    "FCTestModel",
    "MultipleConvTestModel",
    "CustomConvTestModel",
    "CustomConvBNTestModel",
]


class TestManagerForOriginalModels:
    @pytest.fixture(autouse=True, scope="function")
    def init_models(self):
        self.models: Dict[str, ModelDesc] = {
            "ConvBiasBNTestModel": ModelDesc(helpers.ConvBiasBNTestModel, "ConvBiasBNTestModel/Conv2d[conv]/conv2d_0"),
            "ConvBNTestModel": ModelDesc(helpers.ConvBNTestModel, "ConvBNTestModel/Conv2d[conv]/conv2d_0"),
            "ConvTestModel": ModelDesc(helpers.ConvTestModel, "ConvTestModel/Conv2d[conv]/conv2d_0"),
            "FCTestModel": ModelDesc(helpers.FCTestModel, "FCTestModel/Linear[fc]/linear_0"),
            "MultipleConvTestModel": ModelDesc(
                helpers.MultipleConvTestModel, "MultipleConvTestModel/Conv2d[conv_1]/conv2d_0"
            ),
            "CustomConvTestModel": ModelDesc(
                helpers.CustomConvTestModel, "CustomConvTestModel/CustomConv[conv]/conv2d_0"
            ),
            "CustomConvBNTestModel": ModelDesc(
                helpers.CustomConvBNTestModel, "CustomConvBNTestModel/CustomConv[conv]/conv2d_0"
            ),
        }

    REF_FUSED_NODE = {
        "ConvBiasBNTestModel": "ConvBiasBNTestModel/BatchNorm2d[bn]/batch_norm_0",
        "ConvBNTestModel": "ConvBNTestModel/BatchNorm2d[bn]/batch_norm_0",
        "ConvTestModel": None,
        "FCTestModel": None,
        "MultipleConvTestModel": None,
        "CustomConvTestModel": None,
        "CustomConvBNTestModel": "CustomConvBNTestModel/CustomBN2d[bn]/batch_norm_0",
    }

    @pytest.fixture(params=MODELS_LIST)
    def model_desc(self, request) -> Tuple[str, ModelDesc]:
        return request.param, self.models[request.param]

    def test_get_potential_fused_node(self, model_desc):
        model_name, desc = model_desc
        ref = self.REF_FUSED_NODE[model_name]
        fused_node = get_potential_fused_node(desc.node_name, desc.graph)
        result = fused_node.node_name if fused_node is not None else fused_node
        assert result == ref

    REF_WITH_FUSED_BIAS = {
        "ConvBiasBNTestModel": True,
        "ConvBNTestModel": True,
        "ConvTestModel": True,
        "FCTestModel": False,
        "MultipleConvTestModel": True,
        "CustomConvTestModel": True,
        "CustomConvBNTestModel": True,
    }

    def test_is_node_with_fused_bias(self, model_desc):
        model_name, desc = model_desc
        ref = bool(self.REF_WITH_FUSED_BIAS[model_name])
        result = is_node_with_fused_bias(desc.node, desc.graph)
        print(model_name, result)
        assert result == ref

    REF_GET_CONST_NODE = {
        "ConvBiasBNTestModel": ("conv.weight", "conv.bias"),
        "ConvBNTestModel": ("conv.weight", None),
        "ConvTestModel": ("conv.weight", "conv.bias"),
        "FCTestModel": ("fc.weight", "fc.bias"),
        "MultipleConvTestModel": ("conv_1.weight", "conv_1.bias"),
        "CustomConvTestModel": ("conv.weight", "conv.bias"),
        "CustomConvBNTestModel": ("conv.weight", "conv.bias"),
    }

    @pytest.mark.parametrize("port_id", (1, 2))
    def test_get_const_node(self, model_desc, port_id):
        model_name, desc = model_desc
        const_node = get_const_node(desc.node, port_id, desc.graph)
        ref = self.REF_GET_CONST_NODE[model_name][port_id - 1]
        result = const_node.node_name if const_node is not None else const_node
        assert result == ref

    REF_GET_CONST_DATA = {
        "ConvBiasBNTestModel": (
            [[[[0.1000, -2.0000], [1.0000, 0.1000]]], [[[0.1000, 2.0000], [-1.0000, 0.1000]]]],
            [0.3000, 1.3000],
        ),
        "ConvBNTestModel": ([[[[0.1000, -2.0000], [1.0000, 0.1000]]], [[[0.1000, 2.0000], [-1.0000, 0.1000]]]], None),
        "ConvTestModel": (
            [[[[0.1000, -2.0000], [1.0000, 0.1000]]], [[[0.1000, 2.0000], [-1.0000, 0.1000]]]],
            [0.1000, 1.0000],
        ),
        "FCTestModel": ([[0.1000, 0.2000, 0.3000, 0.2000], [0.3000, -0.1000, 0.2000, 0.4000]], [1.0000, 1.1000]),
        "MultipleConvTestModel": (
            [[[[-2.4661, 0.3623], [0.3765, -0.1808]]], [[[0.3930, 0.4327], [-1.3627, 1.3564]]]],
            [0.6688, -0.7077],
        ),
        "CustomConvTestModel": (
            [[[[0.1000, -2.0000], [1.0000, 0.1000]]], [[[0.1000, 2.0000], [-1.0000, 0.1000]]]],
            [0.1000, 1.0000],
        ),
        "CustomConvBNTestModel": (
            [[[[0.1000, -2.0000], [1.0000, 0.1000]]], [[[0.1000, 2.0000], [-1.0000, 0.1000]]]],
            [0.1000, 1.0000],
        ),
    }

    @pytest.mark.parametrize("port_id", (1, 2))
    def test_get_const_data_on_port(self, model_desc, port_id):
        model_name, desc = model_desc
        ref = self.REF_GET_CONST_DATA[model_name][port_id - 1]

        data = get_const_data_on_port(desc.node, port_id, desc.model)
        if ref is None:
            assert data is None
        else:
            assert torch.all(torch.isclose(data, torch.tensor(ref), atol=1e-4))

    REF_WIGHT_PORT_ID = {
        "ConvBiasBNTestModel": [1],
        "ConvBNTestModel": [1],
        "ConvTestModel": [1],
        "FCTestModel": [1],
        "MultipleConvTestModel": [1],
        "CustomConvTestModel": [1],
        "CustomConvBNTestModel": [1],
    }

    def test_get_weight_tensor_port_ids(self, model_desc):
        model_name, desc = model_desc
        result = get_weight_tensor_port_ids(desc.node, desc.graph)
        assert result == self.REF_WIGHT_PORT_ID[model_name]


@pytest.mark.parametrize(
    "const_name, ref",
    (
        ("conv.weight", ("conv", "weight")),
        ("module.head.conv.bias", ("module.head.conv", "bias")),
    ),
    ids=["conv.weight", "module.head.conv.bias"],
)
def test_split_const_name(const_name, ref):
    assert split_const_name(const_name) == ref


class ModelToGet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 1, 1)
        self.customconv = helpers.CustomConv()
        self.seq = nn.Sequential(nn.Identity(), helpers.CustomConv())

    def forward(self, x: torch.Tensor):
        return self.seq(self.conv(x))


def test_get_module_by_name():
    model = ModelToGet()
    assert get_module_by_name("", model) is model
    assert get_module_by_name("conv", model) is model.conv
    assert get_module_by_name("customconv.act", model) is model.customconv.act
    assert get_module_by_name("seq.0", model) is model.seq[0]
    assert get_module_by_name("seq.1", model) is model.seq[1]
    assert get_module_by_name("seq.1.act", model) is model.seq[1].act


def test_get_set_const_data():
    model_cls = helpers.CustomConvBNTestModel
    model = wrap_model(model_cls(), example_input=torch.ones(model_cls.INPUT_SIZE), trace_parameters=True)
    graph = model.nncf.get_graph()
    const_node = graph.get_node_by_name("conv.bias")

    data = get_const_data(const_node, model)
    assert torch.all(model.conv.bias.data == data)
    set_const_data(torch.ones_like(data), const_node, model)
    assert torch.all(model.conv.bias.data == torch.ones_like(data))
