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
from pathlib import Path

import networkx as nx
import pytest

from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph
from tests.cross_fw.shared.nx_graph import check_nx_graph
from tests.cross_fw.shared.paths import TEST_ROOT


@pytest.fixture(scope="module")
def ref_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("Node::A", label=":baz")
    graph.add_node("Node::B", label="qux:")
    graph.add_node("Node::C")
    graph.add_node("D")
    graph.add_node("E", label="no_label")
    graph.add_node("F", label="has^label")
    graph.add_node("F", label="has^label")
    graph.add_edge("Node::A", "Node::B", label="foo:bar"),
    return graph


REF_DOT_REPRESENTATION_GRAPH_PATH = TEST_ROOT / "common" / "data" / "reference_graphs" / "dot_rw_reference.dot"


def test_writing_does_not_modify_original_graph(tmp_path: Path, ref_graph: nx.DiGraph):
    ref_graph_copy = deepcopy(ref_graph)
    write_dot_graph(ref_graph_copy, tmp_path / "graph.dot")
    assert nx.utils.graphs_equal(ref_graph_copy, ref_graph)


def test_colons_are_replaced_in_written_dot_file(tmp_path: Path, ref_graph: nx.DiGraph):
    tmp_path_to_graph = tmp_path / "graph.dot"
    write_dot_graph(ref_graph, tmp_path_to_graph)
    ref = REF_DOT_REPRESENTATION_GRAPH_PATH.read_text()
    act = tmp_path_to_graph.read_text()
    assert ref == act


def test_read_dot_file_gives_graph_with_colons(tmp_path: Path, ref_graph: nx.DiGraph):
    test_graph = read_dot_graph(REF_DOT_REPRESENTATION_GRAPH_PATH)
    check_nx_graph(test_graph, ref_graph, check_edge_attrs=True)
