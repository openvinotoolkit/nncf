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

from collections import Counter

import pytest

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph
from nncf.common.quantization.quantizer_propagation.solver import PostprocessingNodeLocator
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.registry import Registry
from tests.common.quantization.metatypes import WEIGHT_LAYER_METATYPES
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import IdentityTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import NMSTestMetatype
from tests.common.quantization.metatypes import TopKTestMetatype
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.mock_graphs import create_mock_graph
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph

ALL_SYNTHETIC_NNCF_GRAPH = Registry("SYNTHETIC_MODELS")


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest1:
    #              Input_1       Input_2
    #                 |             |
    #               Conv_1          |
    #                 |             |
    #              Identity_1   Identity_3
    #                  |      /     |
    #                  |     /   FC_1
    #                  |    /      |
    #                  NMS_1    Identity_4
    #                   |           |
    #                Identity_2   NMS_2
    #                   |           |
    #                 TopK_1    Identity_5
    #                   |           |
    #                Output_1    Output_2
    #
    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("NMS_1", NMSTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Input_2", InputNoopMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("NMS_2", NMSTestMetatype),
            NodeWithType("Identity_5", IdentityTestMetatype),
            NodeWithType("Output_2", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_1"),
            ("Identity_1", "NMS_1"),
            ("NMS_1", "Identity_2"),
            ("Identity_2", "TopK_1"),
            ("TopK_1", "Output_1"),
            ("Input_2", "Identity_3"),
            ("Identity_3", "NMS_1"),
            ("Identity_3", "FC_1"),
            ("FC_1", "Identity_4"),
            ("Identity_4", "NMS_2"),
            ("NMS_2", "Identity_5"),
            ("Identity_5", "Output_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = [
            "Identity_2",
            "Identity_1",
            "Identity_4",
            "Identity_5",
            "TopK_1",
            "NMS_1",
            "NMS_2",
            "Identity_3",
            "Input_2",
        ]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest2:
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |
    #            TopK_1
    #             |
    #           Identity_2
    #             |
    #           TopK_2
    #             |
    #          Identity_3
    #             |
    #           Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("TopK_2", TopKTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_1"),
            ("Identity_1", "TopK_1"),
            ("TopK_1", "Identity_2"),
            ("Identity_2", "TopK_2"),
            ("TopK_2", "Identity_3"),
            ("Identity_3", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = ["Identity_3", "Identity_2", "Identity_1", "TopK_1", "TopK_2"]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest3:
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |
    #            TopK_1
    #             |
    #           Identity_2
    #             |      \
    #            NMS_1   Conv_2
    #             |        |
    #          Identity_3 Output_2
    #             |
    #           Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("NMS_1", NMSTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Conv_2", Conv2dTestMetatype),
            NodeWithType("Output_2", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_1"),
            ("Identity_1", "TopK_1"),
            ("TopK_1", "Identity_2"),
            ("Identity_2", "NMS_1"),
            ("Identity_2", "Conv_2"),
            ("NMS_1", "Identity_3"),
            ("Identity_3", "Output_1"),
            ("Conv_2", "Output_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = ["Identity_3", "Identity_2", "Identity_1", "NMS_1", "TopK_1"]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest4:
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |       \
    #            TopK_1   Identity_4
    #             |      /       |
    #           Identity_2    Identity_5
    #                \       /
    #                 \     /
    #                  NMS_1
    #                    |
    #                 Identity_3
    #                    |
    #                  Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("NMS_1", NMSTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("Identity_5", IdentityTestMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_1"),
            ("Identity_1", "TopK_1"),
            ("Identity_1", "Identity_4"),
            ("TopK_1", "Identity_2"),
            ("Identity_2", "NMS_1"),
            ("NMS_1", "Identity_3"),
            ("Identity_3", "Output_1"),
            ("Identity_4", "Identity_2"),
            ("Identity_4", "Identity_5"),
            ("Identity_5", "NMS_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = [
            "Identity_3",
            "Identity_2",
            "Identity_5",
            "Identity_4",
            "Identity_1",
            "NMS_1",
            "TopK_1",
        ]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest5:
    #          Input_1
    #             |
    #           NMS_1
    #             \
    #              Conv_1
    #             |       \
    #        Identity_2   Identity_3
    #             |      /       |
    #           Identity_4    Identity_5
    #                \       /
    #                 \     /
    #                  Identity_6
    #                    |
    #                 Identity_7
    #                    |
    #                  Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("NMS_1", IdentityTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("Identity_5", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Identity_6", IdentityTestMetatype),
            NodeWithType("Identity_7", IdentityTestMetatype),
        ]
        node_edges = [
            ("Input_1", "NMS_1"),
            ("NMS_1", "Conv_1"),
            ("Conv_1", "Identity_2"),
            ("Conv_1", "Identity_3"),
            ("Identity_2", "Identity_4"),
            ("Identity_3", "Identity_5"),
            ("Identity_3", "Identity_4"),
            ("Identity_4", "Identity_6"),
            ("Identity_5", "Identity_6"),
            ("Identity_6", "Identity_7"),
            ("Identity_7", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = []


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest6:
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_2
    #             |      \
    #            NMS_1   Conv_2
    #             |        |
    #          Identity_3 Output_2
    #             |
    #           Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("NMS_1", NMSTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Conv_2", Conv2dTestMetatype),
            NodeWithType("Output_2", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_2"),
            ("Identity_2", "NMS_1"),
            ("Identity_2", "Conv_2"),
            ("NMS_1", "Identity_3"),
            ("Identity_3", "Output_1"),
            ("Conv_2", "Output_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = ["Identity_3", "Identity_2", "NMS_1"]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest7:
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |       \
    #            TopK_1   Identity_4
    #             |      /       |
    #           Identity_2    Identity_5
    #                \       /
    #                 \     /
    #                 Identity_3
    #                    |
    #                  Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("Identity_5", IdentityTestMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_1"),
            ("Identity_1", "TopK_1"),
            ("Identity_1", "Identity_4"),
            ("TopK_1", "Identity_2"),
            ("Identity_2", "Identity_3"),
            ("Identity_3", "Output_1"),
            ("Identity_4", "Identity_2"),
            ("Identity_4", "Identity_5"),
            ("Identity_5", "Identity_3"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = ["Identity_3", "Identity_2", "Identity_1", "TopK_1"]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest8:
    #               Input_1      Input_2
    #                  |            |
    #                Conv_1      Conv_2
    #                  |            |
    #              Identity_1   Identity_3
    #                  |      /     |
    #                  |     /    FC_1
    #                  |    /       |
    #                  NMS_1    Identity_4
    #                   |           |
    #                Identity_2 Identity_5
    #                   |           |
    #                 TopK_1    Output_2
    #                   |
    #                Output_1
    #
    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("NMS_1", NMSTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType("Input_2", InputNoopMetatype),
            NodeWithType("Conv_2", Conv2dTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("Identity_5", IdentityTestMetatype),
            NodeWithType("Output_2", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Identity_1"),
            ("Identity_1", "NMS_1"),
            ("NMS_1", "Identity_2"),
            ("Identity_2", "TopK_1"),
            ("TopK_1", "Output_1"),
            ("Input_2", "Conv_2"),
            ("Conv_2", "Identity_3"),
            ("Identity_3", "NMS_1"),
            ("Identity_3", "FC_1"),
            ("FC_1", "Identity_4"),
            ("Identity_4", "Identity_5"),
            ("Identity_5", "Output_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = ["Identity_1", "Identity_2", "Identity_3", "TopK_1", "NMS_1"]


@ALL_SYNTHETIC_NNCF_GRAPH.register()
class ModelToTest9:
    #          Input_1
    #             |
    #         Identity_1
    #             |
    #           TopK_1
    #             |
    #         Identity_2
    #             |
    #          Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("TopK_1", TopKTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Identity_1"),
            ("Identity_1", "TopK_1"),
            ("TopK_1", "Identity_2"),
            ("Identity_2", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.reference_ignored_scopes = ["Input_1", "Identity_1", "TopK_1", "Identity_2"]


@pytest.mark.parametrize("model_to_test", ALL_SYNTHETIC_NNCF_GRAPH.values())
def test_node_locator_finds_postprocessing_nodes(model_to_test):
    model_to_test = model_to_test()
    nncf_graph = model_to_test.nncf_graph

    ip_graph = InsertionPointGraph(nncf_graph)

    weight_nodes = nncf_graph.get_nodes_by_metatypes(WEIGHT_LAYER_METATYPES)
    quantizable_layer_nodes = [
        QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in weight_nodes
    ]

    quant_prop_graph = QuantizerPropagationStateGraph(ip_graph)
    post_processing_node_locator = PostprocessingNodeLocator(
        quant_prop_graph, quantizable_layer_nodes, [TopKTestMetatype, NMSTestMetatype]
    )
    ignored_node_keys = post_processing_node_locator.get_post_processing_node_keys()

    ignored_node_names = [
        nncf_graph.get_node_by_key(ignored_node_key).node_type for ignored_node_key in ignored_node_keys
    ]
    assert Counter(ignored_node_names) == Counter(model_to_test.reference_ignored_scopes)
