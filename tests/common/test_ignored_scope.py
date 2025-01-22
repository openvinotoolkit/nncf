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

import pytest

import nncf
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.scopes import IgnoredScope
from nncf.scopes import Subgraph
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.mock_graphs import create_mock_graph
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph

LINEAR_TYPE = "linear"
CONV_TYPE = "conv"


class NNCFGraphToTestIgnoredScope:
    def __init__(self, conv_metatype, linear_metatype):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype, op_type=conv_metatype),
            NodeWithType("Linear_1", LinearTestMetatype, op_type=linear_metatype),
            NodeWithType("Conv_2", Conv2dTestMetatype, op_type=conv_metatype),
            NodeWithType("Linear_2", LinearTestMetatype, op_type=linear_metatype),
            NodeWithType("Marked_Conv_3", Conv2dTestMetatype, op_type=conv_metatype),
            NodeWithType("Linear_3", LinearTestMetatype, op_type=linear_metatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Linear_1"),
            ("Linear_1", "Conv_2"),
            ("Conv_2", "Linear_2"),
            ("Linear_2", "Marked_Conv_3"),
            ("Marked_Conv_3", "Linear_3"),
            ("Linear_3", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)


IGNORED_SCOPES_TEST_DATA = [
    (IgnoredScope(), []),
    (IgnoredScope(["/Conv_1_0", "/Linear_1_0"]), ["/Conv_1_0", "/Linear_1_0"]),
    (IgnoredScope(["/Conv_1_0", "/Linear_1_0"], [".*Marked.*"]), ["/Conv_1_0", "/Linear_1_0", "/Marked_Conv_3_0"]),
    (
        IgnoredScope(["/Conv_1_0", "/Linear_1_0"], [".*Marked.*"], [LINEAR_TYPE]),
        ["/Conv_1_0", "/Linear_1_0", "/Linear_2_0", "/Linear_3_0", "/Marked_Conv_3_0"],
    ),
    (
        IgnoredScope(subgraphs=[Subgraph(inputs=["/Linear_1_0"], outputs=["/Linear_3_0"])]),
        ["/Conv_2_0", "/Linear_1_0", "/Linear_2_0", "/Linear_3_0", "/Marked_Conv_3_0"],
    ),
    (
        IgnoredScope(subgraphs=[Subgraph(inputs=["/Linear_1_0"], outputs=["/Linear_1_0"])]),
        ["/Linear_1_0"],
    ),
]


@pytest.mark.parametrize("ignored_scope,ref_ignored_names", IGNORED_SCOPES_TEST_DATA)
def test_ignored_scopes(ignored_scope, ref_ignored_names):
    nncf_graph = NNCFGraphToTestIgnoredScope(CONV_TYPE, LINEAR_TYPE).nncf_graph
    ignored_names = get_ignored_node_names_from_ignored_scope(ignored_scope, nncf_graph)
    assert sorted(ignored_names) == ref_ignored_names


WRONG_IGNORED_SCOPES_TEST_DATA = [
    IgnoredScope(["/Conv_0_0", "/Conv_1_0", "/Linear_1_0"]),
    IgnoredScope(patterns=[".*Maarked.*"]),
    IgnoredScope(types=["wrong_type"]),
    IgnoredScope(subgraphs=[Subgraph(inputs=["/Linear_3_0"], outputs=["/Linear_1_0"])]),
]


@pytest.mark.parametrize("ignored_scope", WRONG_IGNORED_SCOPES_TEST_DATA)
def test_wrong_ignored_scopes(ignored_scope):
    nncf_graph = NNCFGraphToTestIgnoredScope(CONV_TYPE, LINEAR_TYPE).nncf_graph
    with pytest.raises(nncf.ValidationError):
        get_ignored_node_names_from_ignored_scope(ignored_scope, nncf_graph)
