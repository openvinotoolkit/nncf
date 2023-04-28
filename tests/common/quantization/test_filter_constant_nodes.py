"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import re
from collections import Counter

import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.insertion_point_graph import ConstantNodesFilter
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.registry import Registry
from tests.common.quantization.metatypes import WEIGHT_LAYER_METATYPES
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import IdentityTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import ReshapeTestMetatype
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.mock_graphs import create_mock_graph
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph

SYNTHETIC_NNCF_GRAPH_WITH_CONSTANT_SUBGRAPHS = Registry("SYNTHETIC_MODELS_WITH_CONSTANT_SUBGRAPHS")


@SYNTHETIC_NNCF_GRAPH_WITH_CONSTANT_SUBGRAPHS.register()
class ModelToTest1:
    #       Original graph                            Filtered graph
    #          Input_1        Reshape_1                   Input_1
    #             |          /                              |
    #           Conv_1  Identity_1                        Conv_1
    #             |    /                                    |
    #            FC_1                                      FC_1
    #             |                                         |
    #          Identity_2                               Identity_2
    #             |                                         |
    #           FC_2 --- Identity_3                        FC_2
    #             |                                         |
    #           Output_1                                 Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("Reshape_1", ReshapeTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("FC_2", LinearTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "FC_1"),
            ("Identity_1", "FC_1"),
            ("Reshape_1", "Identity_1"),
            ("FC_1", "Identity_2"),
            ("Identity_2", "FC_2"),
            ("Identity_3", "FC_2"),
            ("FC_2", "Output_1"),
        ]
        ref_nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("FC_2", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        ref_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "FC_1"),
            ("FC_1", "Identity_2"),
            ("Identity_2", "FC_2"),
            ("FC_2", "Output_1"),
        ]

        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        reference_mock_graph = create_mock_graph(ref_nodes, ref_edges)
        self.ref_nncf_graph = get_nncf_graph_from_mock_nx_graph(reference_mock_graph)


@SYNTHETIC_NNCF_GRAPH_WITH_CONSTANT_SUBGRAPHS.register()
class ModelToTest2:
    #       Original graph                            Filtered graph
    #          Input_1        Conv_1                    Input_1     Conv_1
    #             |          /                              |       /
    #           Conv_2  Identity_1                        Conv_2  Identity_1
    #             |    /                                    |   /
    #            FC_1                                      FC_1
    #             |                                         |
    #          Identity_2                               Identity_2
    #             |                                         |
    #           FC_2 --- Identity_3                        FC_2
    #             |                                         |
    #           Output_1                                 Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_2", Conv2dTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("FC_2", LinearTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_2", "FC_1"),
            ("Identity_1", "FC_1"),
            ("Conv_1", "Identity_1"),
            ("FC_1", "Identity_2"),
            ("Identity_2", "FC_2"),
            ("Identity_3", "FC_2"),
            ("FC_2", "Output_1"),
        ]
        ref_nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_2", Conv2dTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("FC_2", LinearTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        ref_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_2", "FC_1"),
            ("Identity_1", "FC_1"),
            ("Conv_1", "Identity_1"),
            ("FC_1", "Identity_2"),
            ("Identity_2", "FC_2"),
            ("FC_2", "Output_1"),
        ]

        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        reference_mock_graph = create_mock_graph(ref_nodes, ref_edges)
        self.ref_nncf_graph = get_nncf_graph_from_mock_nx_graph(reference_mock_graph)


@SYNTHETIC_NNCF_GRAPH_WITH_CONSTANT_SUBGRAPHS.register()
class ModelToTest3:
    #       Original graph                            Filtered graph
    #                               (Graph will not be filtered, because there is no Input node)
    #          Identity_1                               Identity_1
    #             |                                        |
    #           Conv_1  Identity_2                       Conv_1  Identity_2
    #             |    /                                   |    /
    #            FC_1                                     FC_1
    #             |                                        |
    #          Identity_3                               Identity_3
    #             |                                        |
    #           FC_2 --- Identity_4                      FC_2 --- Identity_4
    #             |                                        |
    #           Output_1                                 Output_1

    def __init__(self):
        nodes = [
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("FC_2", LinearTestMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Identity_1", "Conv_1"),
            ("Conv_1", "FC_1"),
            ("Identity_2", "FC_1"),
            ("FC_1", "Identity_3"),
            ("Identity_3", "FC_2"),
            ("Identity_4", "FC_2"),
            ("FC_2", "Output_1"),
        ]
        ref_nodes = [
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("Conv_1", Conv2dTestMetatype),
            NodeWithType("FC_1", LinearTestMetatype),
            NodeWithType("Identity_2", IdentityTestMetatype),
            NodeWithType("Identity_3", IdentityTestMetatype),
            NodeWithType("FC_2", LinearTestMetatype),
            NodeWithType("Identity_4", IdentityTestMetatype),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        ref_edges = [
            ("Identity_1", "Conv_1"),
            ("Conv_1", "FC_1"),
            ("Identity_2", "FC_1"),
            ("FC_1", "Identity_3"),
            ("Identity_3", "FC_2"),
            ("Identity_4", "FC_2"),
            ("FC_2", "Output_1"),
        ]

        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        reference_mock_graph = create_mock_graph(ref_nodes, ref_edges)
        self.ref_nncf_graph = get_nncf_graph_from_mock_nx_graph(reference_mock_graph)


@pytest.mark.parametrize("model_to_test", SYNTHETIC_NNCF_GRAPH_WITH_CONSTANT_SUBGRAPHS.values())
def test_constant_nodes_filter(model_to_test):
    model_to_test = model_to_test()
    nncf_graph = model_to_test.nncf_graph
    weight_nodes = nncf_graph.get_nodes_by_metatypes(WEIGHT_LAYER_METATYPES)
    quantizable_layer_nodes = [
        QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in weight_nodes
    ]
    quantizable_layer_node_keys = [node.node.data[NNCFGraph.KEY_NODE_ATTR] for node in quantizable_layer_nodes]

    ip_graph = get_ip_graph_for_test(nncf_graph, quantizable_layer_nodes)
    filtered_ip_graph = ConstantNodesFilter.filter(ip_graph, quantizable_layer_node_keys)

    ref_ip_graph = get_ip_graph_for_test(model_to_test.ref_nncf_graph, quantizable_layer_nodes)

    check_ip_graphs_are_equal(filtered_ip_graph, ref_ip_graph)


def check_ip_graphs_are_equal(graph_1: InsertionPointGraph, graph_2: InsertionPointGraph):
    graph_1_node_keys_without_index = [graph_1_node_key.split(" ")[-1] for graph_1_node_key in graph_1.nodes.keys()]
    graph_2_node_keys_without_index = [graph_2_node_key.split(" ")[-1] for graph_2_node_key in graph_2.nodes.keys()]
    assert Counter(graph_1_node_keys_without_index) == Counter(graph_2_node_keys_without_index)

    graph_1_filtered_edges, graph_2_filtered_edges = [], []
    for edge in graph_1.edges:
        graph_1_filtered_edges.append((filter_edge(edge[0]), filter_edge(edge[1])))
    for edge in graph_2.edges:
        graph_2_filtered_edges.append((filter_edge(edge[0]), filter_edge(edge[1])))
    assert Counter(graph_1_filtered_edges) == Counter(graph_2_filtered_edges)


def filter_edge(edge: str) -> str:
    """
    Removes node ids from the edges.

    :param edge: Edges to remove node ids.
    :return: Filtered edge.
    """
    splitted_edge = edge.split(" ")
    filtered_edge = []
    for word in splitted_edge:
        if re.match("[0-9]+", word) is None:
            filtered_edge.append(word)
    return "".join(filtered_edge)
