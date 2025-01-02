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

from copy import deepcopy
from dataclasses import dataclass
from typing import List

import pytest

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut
from nncf.quantization.passes import find_shapeof_subgraphs
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
            Node(119, "fake_quantize"),
            Node(124, "relu"),
            Node(125, "add"),
            Node(126, "constant"),
            Node(127, "conv2d"),
            Node(128, "fake_quantize"),
            Node(133, "constant"),
            Node(134, "fake_quantize"),
            Node(139, "fake_quantize"),
            Node(144, "add"),
            Node(145, "constant"),
            Node(146, "conv2d"),
            Node(147, "fake_quantize"),
            Node(152, "constant"),
            Node(153, "fake_quantize"),
            Node(158, "relu"),
            Node(159, "add"),
            Node(160, "constant"),
            Node(161, "conv2d"),
            Node(162, "fake_quantize"),
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
            Node(82, "fake_quantize"),
            Node(720, "constant"),
            Node(87, "power"),
            Node(93, "fake_quantize"),
            Node(99, "multiply"),
            Node(710, "fake_quantize"),
            Node(715, "constant"),
            Node(106, "shapeof"),
            Node(105, "fake_quantize"),
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
    "simple_graph_quantize_dequantize": Graph(
        [
            Node(0, "parameter"),
            Node(37, "quantize"),
            Node(38, "dequantize"),
            Node(41, "conv2d"),
            Node(42, "quantize"),
            Node(40, "dequantize"),
            Node(43, "dequantize"),
            Node(39, "quantize"),
            Node(46, "conv2d"),
            Node(65, "add"),
            Node(45, "dequantize"),
            Node(64, "dequantize"),
            Node(44, "quantize"),
            Node(63, "quantize"),
        ],
        [
            Edge(0, 0, 37, 0),
            Edge(37, 0, 38, 0),
            Edge(38, 0, 41, 0),
            Edge(41, 0, 42, 0),
            Edge(42, 0, 43, 0),
            Edge(40, 0, 41, 1),
            Edge(43, 0, 46, 0),
            Edge(43, 0, 65, 0),
            Edge(39, 0, 40, 0),
            Edge(45, 0, 46, 1),
            Edge(64, 0, 65, 1),
            Edge(44, 0, 45, 0),
            Edge(63, 0, 64, 0),
            Edge(46, 0, 63, 0),
        ],
    ),
}


@dataclass
class ParameterTestCase:
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
        ParameterTestCase(
            "fake_quantize_119",
            ["fake_quantize_139", "fake_quantize_162", "fake_quantize_119"],
            ["add_117", "conv2d_161"],
        ),
        ParameterTestCase("fake_quantize_128", ["fake_quantize_134", "fake_quantize_128"], ["conv2d_127"]),
        ParameterTestCase("fake_quantize_134", ["fake_quantize_134", "fake_quantize_128"], ["conv2d_127"]),
        ParameterTestCase(
            "fake_quantize_139",
            ["fake_quantize_139", "fake_quantize_162", "fake_quantize_119"],
            ["add_117", "conv2d_161"],
        ),
        ParameterTestCase("fake_quantize_147", ["fake_quantize_153", "fake_quantize_147"], ["conv2d_146"]),
        ParameterTestCase("fake_quantize_153", ["fake_quantize_153", "fake_quantize_147"], ["conv2d_146"]),
        ParameterTestCase(
            "fake_quantize_162",
            ["fake_quantize_139", "fake_quantize_162", "fake_quantize_119"],
            ["add_117", "conv2d_161"],
        ),
    ],
    "graph_with_shapeof": [ParameterTestCase("fake_quantize_105", ["fake_quantize_105"], ["interpolate_115"])],
    "simple_graph_quantize_dequantize": [
        ParameterTestCase(
            "quantize_37", ["quantize_37", "dequantize_38", "quantize_39", "dequantize_40"], ["conv2d_41"]
        ),
        ParameterTestCase(
            "quantize_39", ["quantize_37", "dequantize_38", "quantize_39", "dequantize_40"], ["conv2d_41"]
        ),
        #
        ParameterTestCase(
            "quantize_42",
            ["quantize_42", "dequantize_43", "quantize_44", "dequantize_45", "quantize_63", "dequantize_64"],
            ["conv2d_46", "add_65"],
        ),
        ParameterTestCase(
            "quantize_44",
            ["quantize_42", "dequantize_43", "quantize_44", "dequantize_45", "quantize_63", "dequantize_64"],
            ["conv2d_46", "add_65"],
        ),
        ParameterTestCase(
            "quantize_63",
            ["quantize_42", "dequantize_43", "quantize_44", "dequantize_45", "quantize_63", "dequantize_64"],
            ["conv2d_46", "add_65"],
        ),
    ],
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
    ids = []
    for graph_name, test_cases in TEST_CASES.items():
        nncf_graph = create_nncf_graph(GRAPHS[graph_name])
        for i, test_case in enumerate(test_cases):
            ids.append(f"{graph_name}_{i}")
            test_params.append((nncf_graph, test_case))
    return ids, test_params


IDS, TEST_PARAMS = create_test_params()


@pytest.mark.parametrize("nncf_graph,test_case", TEST_PARAMS, ids=IDS)
def test_find_quantizer_nodes_to_cut(nncf_graph: NNCFGraph, test_case: ParameterTestCase):
    quantizer_node = nncf_graph.get_node_by_name(test_case.node_name)
    # As test graphs are fully connected and does not have readvariable metatype,
    # this should work
    input_nodes = nncf_graph.get_input_nodes()

    shapeof_subgraphs = find_shapeof_subgraphs(nncf_graph, SHAPEOF_METATYPES, input_nodes)
    nncf_graph_without_shapeof = deepcopy(nncf_graph)
    nncf_graph_without_shapeof.remove_nodes_from(shapeof_subgraphs)

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
