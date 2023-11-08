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
from pathlib import Path

import networkx as nx
import pytest

from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.common.utils.os import safe_open


@pytest.fixture(scope="module")
def ref_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("Node::A", label=":baz")
    graph.add_node("Node::B", label="qux:")
    graph.add_edge("Node::A", "Node::B", label="foo:bar")
    return graph


REF_DOT_REPRESENTATION = """strict digraph  {
"Node^^A" [label="^baz"];
"Node^^B" [label="qux^"];
"Node^^A" -> "Node^^B"  [label="foo^bar"];
}
"""


def test_writing_does_not_modify_original_graph(tmp_path: Path, ref_graph: nx.DiGraph):
    ref_graph_copy = deepcopy(ref_graph)
    write_dot_graph(ref_graph_copy, tmp_path / "graph.dot")
    assert nx.utils.graphs_equal(ref_graph_copy, ref_graph)


def test_colons_are_replaced_in_written_dot_file(tmp_path: Path, ref_graph: nx.DiGraph):
    tmp_path_to_graph = tmp_path / "graph.dot"
    write_dot_graph(ref_graph, tmp_path_to_graph)
    with safe_open(tmp_path_to_graph, "r", encoding="utf-8") as f:
        dot_contents = "\n".join(f.read().splitlines())
        assert dot_contents == "\n".join(REF_DOT_REPRESENTATION.splitlines())
