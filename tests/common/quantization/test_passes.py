# Copyright (c) 2023 Intel Corporation
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
from typing import Dict

import pytest

from nncf.common.graph import NNCFGraph
from nncf.quantization.passes import filter_constant_nodes
from nncf.quantization.passes import insert_noops_instead_constants
from nncf.quantization.passes import remove_shapeof_subgraphs
from tests.common.quantization.entities import Edge
from tests.common.quantization.entities import Graph
from tests.common.quantization.entities import Node
from tests.common.quantization.metatypes import READ_VARIABLE_METATYPES
from tests.common.quantization.metatypes import SHAPEOF_METATYPES
from tests.common.quantization.metatypes import AddTestMetatype
from tests.common.quantization.metatypes import GatherTestMetatype
from tests.common.quantization.mock_graphs import create_nncf_graph
from tests.shared.nx_graph import compare_nx_graph_with_reference
from tests.shared.paths import TEST_ROOT

REFERENCES_PATH = TEST_ROOT / "common" / "data"


@dataclass
class TestCaseData:
    name: str
    methods_with_kwargs: Dict[callable, Dict]


def graph_with_shapeof() -> NNCFGraph:
    graph = Graph(
        [
            Node(0, "parameter"),
            Node(1, "power"),
            Node(2, "constant"),
            Node(3, "multiply"),
            Node(4, "constant"),
            Node(5, "shapeof"),
            Node(6, "interpolate"),
            Node(7, "strided_slice"),
            Node(8, "constant"),
            Node(9, "constant"),
            Node(10, "constant"),
            Node(11, "concat"),
            Node(12, "constant"),
            Node(13, "convert"),
            Node(14, "convert"),
            Node(15, "divide"),
            Node(16, "constant"),
            Node(17, "add"),
            Node(18, "add"),
            Node(19, "read_value"),
            Node(20, "constant"),
        ],
        [
            Edge(0, 0, 1, 0),
            Edge(2, 0, 1, 1),
            Edge(1, 0, 3, 0),
            Edge(4, 0, 3, 1),
            Edge(3, 0, 6, 0),
            Edge(3, 0, 5, 0),
            Edge(5, 0, 7, 0),
            Edge(8, 0, 7, 1),
            Edge(9, 0, 7, 2),
            Edge(10, 0, 7, 3),
            Edge(7, 0, 11, 0),
            Edge(12, 0, 11, 1),
            Edge(11, 0, 6, 1),
            Edge(11, 0, 13, 0),
            Edge(13, 0, 15, 0),
            Edge(15, 0, 17, 0),
            Edge(16, 0, 17, 1),
            Edge(17, 0, 6, 2),
            Edge(5, 0, 14, 0),
            Edge(14, 0, 15, 1),
            Edge(6, 0, 18, 0),
            Edge(20, 0, 19, 0),
            Edge(19, 0, 18, 1),
        ],
    )
    return create_nncf_graph(graph)


def graph_with_constants() -> NNCFGraph:
    graph = Graph(
        [
            Node(0, "parameter"),
            Node(1, "multiply"),
            Node(2, "constant"),
            Node(3, "add"),
            Node(4, "constant"),
            Node(5, "conv2d"),
            Node(6, "constant"),
            Node(7, "add"),
            Node(8, "constant"),
            Node(9, "conv2d"),
            Node(10, "constant"),
            Node(11, "relu"),
            Node(12, "matmul"),
            Node(13, "constant"),
            Node(14, "matmul"),
            Node(15, "concat"),
            Node(16, "convert"),
            Node(17, "gather"),
            Node(18, "constant"),
            Node(19, "add"),
            Node(20, "read_value"),
            Node(21, "constant"),
            Node(22, "conv2d"),
            Node(23, "constant"),
            Node(24, "reshape"),
            Node(25, "constant"),
        ],
        [
            Edge(0, 0, 1, 0),
            Edge(2, 0, 1, 1),
            Edge(1, 0, 3, 0),
            Edge(4, 0, 3, 1),
            Edge(3, 0, 5, 0),
            Edge(6, 0, 5, 1),
            Edge(5, 0, 7, 0),
            Edge(8, 0, 7, 1),
            Edge(3, 0, 9, 0),
            Edge(10, 0, 9, 1),
            Edge(9, 0, 11, 0),
            Edge(11, 0, 12, 1),
            Edge(13, 0, 12, 0),
            Edge(12, 0, 14, 0),
            Edge(7, 0, 14, 1),
            Edge(14, 0, 15, 0),
            Edge(9, 0, 15, 1),
            Edge(3, 0, 15, 2),
            Edge(15, 0, 16, 0),
            Edge(16, 0, 17, 0),
            Edge(18, 0, 17, 1),
            Edge(17, 0, 19, 1),
            Edge(20, 0, 19, 0),
            Edge(21, 0, 20, 0),
            Edge(19, 0, 22, 0),
            Edge(23, 0, 22, 1),
            Edge(22, 0, 24, 0),
            Edge(25, 0, 24, 1),
        ],
    )
    return create_nncf_graph(graph)


def compare_nncf_graphs(nncf_graph: NNCFGraph, path_ref_graph: str):
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    compare_nx_graph_with_reference(nx_graph, path_ref_graph, check_edge_attrs=True, unstable_node_names=True)


TEST_CONFIGS = [
    (
        graph_with_shapeof(),
        TestCaseData(
            "graph_without_shapeof",
            {
                remove_shapeof_subgraphs: {
                    "shapeof_metatypes": SHAPEOF_METATYPES,
                    "read_variable_metatypes": READ_VARIABLE_METATYPES,
                }
            },
        ),
    ),
    (
        graph_with_constants(),
        TestCaseData(
            "graph_without_constants", {filter_constant_nodes: {"read_variable_metatypes": READ_VARIABLE_METATYPES}}
        ),
    ),
    (
        graph_with_constants(),
        TestCaseData(
            "graph_with_noops",
            {
                filter_constant_nodes: {"read_variable_metatypes": READ_VARIABLE_METATYPES},
                insert_noops_instead_constants: {
                    "original_nncf_graph": graph_with_constants(),
                    "metatypes_to_insert_noop": [GatherTestMetatype, AddTestMetatype],
                },
            },
        ),
    ),
]


@pytest.mark.parametrize(
    "nncf_graph, test_case", TEST_CONFIGS, ids=["{}".format(data.name) for _, data in TEST_CONFIGS]
)
def test_pass(nncf_graph: NNCFGraph, test_case: TestCaseData):
    processed_graph = deepcopy(nncf_graph)
    for method, kwargs in test_case.methods_with_kwargs.items():
        processed_graph = method(processed_graph, **kwargs)

    compare_nncf_graphs(processed_graph, f"{REFERENCES_PATH}/{test_case.name}.dot")
