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

from copy import deepcopy
from dataclasses import dataclass
from typing import List

import pytest

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut
from nncf.quantization.passes import remove_shapeof_subgraphs
from tests.common.quantization.metatypes import CONSTANT_METATYPES
from tests.common.quantization.metatypes import METATYPES_FOR_TEST
from tests.common.quantization.metatypes import QUANTIZABLE_METATYPES
from tests.common.quantization.metatypes import QUANTIZE_AGNOSTIC_METATYPES
from tests.common.quantization.metatypes import QUANTIZER_METATYPES
from tests.common.quantization.metatypes import SHAPEOF_METATYPES


@dataclass
class Node:
    node_id: int
    node_type: str


@dataclass
class Edge:
    from_node_id: int
    from_port: int
    to_node_id: int
    to_port: int


@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Edge]


GRAPHS = {
    "simple_graph": Graph(
        [
            Node(0, "parameter"),
            Node(116, "relu"),
            Node(117, "add"),
            Node(118, "max_pool2d"),
            Node(119, "quantizer"),
            Node(124, "relu"),
            Node(125, "add"),
            Node(126, "constant"),
            Node(127, "conv2d"),
            Node(128, "quantizer"),
            Node(133, "constant"),
            Node(134, "quantizer"),
            Node(139, "quantizer"),
            Node(144, "add"),
            Node(145, "constant"),
            Node(146, "conv2d"),
            Node(147, "quantizer"),
            Node(152, "constant"),
            Node(153, "quantizer"),
            Node(158, "relu"),
            Node(159, "add"),
            Node(160, "constant"),
            Node(161, "conv2d"),
            Node(162, "quantizer"),
            Node(167, "constant"),
        ],
        [
            Edge(117, 0, 116, 0),
            Edge(118, 0, 117, 1),
            Edge(139, 0, 117, 0),
            Edge(119, 0, 118, 0),
            Edge(124, 0, 119, 0),
            Edge(125, 0, 124, 0),
            Edge(126, 0, 125, 1),
            Edge(127, 0, 125, 0),
            Edge(128, 0, 127, 1),
            Edge(134, 0, 127, 0),
            Edge(133, 0, 128, 0),
            Edge(0, 0, 134, 0),
            Edge(144, 0, 139, 0),
            Edge(145, 0, 144, 1),
            Edge(146, 0, 144, 0),
            Edge(147, 0, 146, 1),
            Edge(153, 0, 146, 0),
            Edge(152, 0, 147, 0),
            Edge(158, 0, 153, 0),
            Edge(159, 0, 158, 0),
            Edge(160, 0, 159, 1),
            Edge(161, 0, 159, 0),
            Edge(118, 0, 161, 0),
            Edge(162, 0, 161, 1),
            Edge(167, 0, 162, 0),
        ],
    ),
    "graph_with_shapeof": Graph(
        [
            Node(0, "parameter"),
            Node(82, "quantizer"),
            Node(720, "constant"),
            Node(87, "power"),
            Node(93, "quantizer"),
            Node(99, "multiply"),
            Node(710, "quantizer"),
            Node(715, "constant"),
            Node(106, "shapeof"),
            Node(105, "quantizer"),
            Node(115, "interpolate"),
            Node(117, "strided_slice"),
            Node(746, "constant"),
            Node(745, "constant"),
            Node(744, "constant"),
            Node(130, "concat"),
            Node(647, "constant"),
            Node(116, "convert"),
            Node(142, "convert"),
            Node(129, "divide"),
            Node(709, "constant"),
            Node(141, "add"),
        ],
        [
            Edge(82, 0, 87, 0),
            Edge(720, 0, 87, 1),
            Edge(87, 0, 93, 0),
            Edge(93, 0, 99, 0),
            Edge(99, 0, 105, 0),
            Edge(99, 0, 106, 0),
            Edge(710, 0, 99, 1),
            Edge(715, 0, 710, 0),
            Edge(106, 0, 117, 0),
            Edge(106, 0, 116, 0),
            Edge(105, 0, 115, 0),
            Edge(117, 0, 130, 0),
            Edge(746, 0, 117, 1),
            Edge(745, 0, 117, 2),
            Edge(744, 0, 117, 3),
            Edge(130, 0, 142, 0),
            Edge(130, 0, 115, 1),
            Edge(647, 0, 130, 1),
            Edge(116, 0, 129, 1),
            Edge(142, 0, 129, 0),
            Edge(129, 0, 141, 0),
            Edge(709, 0, 141, 1),
            Edge(141, 0, 115, 2),
            Edge(0, 0, 82, 0),
        ],
    ),
}


@dataclass
class TestCase:
    """
    :param node_name: Quantizer node's name. We want to remove this
        quantizer from the model.
    :param nodes: Expected list of quantizer nodes that should be removed
        from the model. It includes `node_name` and additional quantizer
        node's name
    :param ops: Expected list of operations that will be reverted to
        original precision after removing `nodes`.
    """

    node_name: str
    nodes: List[str]
    ops: List[str]


TEST_CASES = {
    "simple_graph": [
        TestCase("quantizer_119", ["quantizer_139", "quantizer_162", "quantizer_119"], ["add_117", "conv2d_161"]),
        TestCase("quantizer_128", ["quantizer_134", "quantizer_128"], ["conv2d_127"]),
        TestCase("quantizer_134", ["quantizer_134", "quantizer_128"], ["conv2d_127"]),
        TestCase("quantizer_139", ["quantizer_139", "quantizer_162", "quantizer_119"], ["add_117", "conv2d_161"]),
        TestCase("quantizer_147", ["quantizer_153", "quantizer_147"], ["conv2d_146"]),
        TestCase("quantizer_153", ["quantizer_153", "quantizer_147"], ["conv2d_146"]),
        TestCase("quantizer_162", ["quantizer_139", "quantizer_162", "quantizer_119"], ["add_117", "conv2d_161"]),
    ],
    "graph_with_shapeof": [TestCase("quantizer_105", ["quantizer_105"], ["interpolate_115"])],
}


def create_nncf_graph(graph: Graph) -> NNCFGraph:
    """
    Creates NNCF graph from provided graph's description.

    :param graph: Graph's description.
    :return: NNCFGraph instance.
    """
    nncf_graph = NNCFGraph()

    correspondences = {}
    for v in graph.nodes:
        node_name = f"{v.node_type}_{v.node_id}"
        metatype = METATYPES_FOR_TEST.get_operator_metatype_by_op_name(v.node_type)
        node = nncf_graph.add_nncf_node(node_name, v.node_type, metatype)
        correspondences[v.node_id] = node.node_id

    for e in graph.edges:
        nncf_graph.add_edge_between_nncf_nodes(
            correspondences[e.from_node_id], correspondences[e.to_node_id], [], e.to_port, e.from_port, Dtype.FLOAT
        )

    return nncf_graph


def create_test_params():
    test_params = []
    for graph_name, test_cases in TEST_CASES.items():
        nncf_graph = create_nncf_graph(GRAPHS[graph_name])
        for test_case in test_cases:
            test_params.append((nncf_graph, test_case))
    return test_params


@pytest.mark.parametrize("nncf_graph,test_case", create_test_params())
def test_find_quantizer_nodes_to_cut(nncf_graph: NNCFGraph, test_case: TestCase):
    quantizer_node = nncf_graph.get_node_by_name(test_case.node_name)
    nncf_graph_without_shapeof = remove_shapeof_subgraphs(deepcopy(nncf_graph), SHAPEOF_METATYPES)
    nodes, ops = find_quantizer_nodes_to_cut(
        nncf_graph_without_shapeof,
        quantizer_node,
        QUANTIZER_METATYPES,
        CONSTANT_METATYPES,
        QUANTIZABLE_METATYPES,
        QUANTIZE_AGNOSTIC_METATYPES,
    )

    actual_fq_nodes = sorted([x.node_name for x in nodes])
    actual_ops = sorted([x.node_name for x in ops])

    assert actual_fq_nodes == sorted(test_case.nodes)
    assert actual_ops == sorted(test_case.ops)
