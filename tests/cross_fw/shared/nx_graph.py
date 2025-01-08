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

import os
import re
from functools import total_ordering
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import networkx as nx

import nncf
from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph


def sort_dot(path):
    with open(path, "r", encoding="utf8") as f:
        content = f.readlines()
    start_line = "strict digraph  {\n"
    end_line = "}\n"
    content.remove(start_line)
    content.remove(end_line)

    @total_ordering
    class LineOrder:
        """Helper structure to sort node lines before edge lines in .dot graphs, and within the edge lines -
        edges with lower starting IDs first, for edges with identical starting IDs - edges with lower ending IDs
        first."""

        def __init__(
            self, node_id: Optional[int] = None, edge_start_id: Optional[int] = None, edge_end_id: Optional[int] = None
        ):
            if node_id is not None:
                if edge_start_id is not None or edge_end_id is not None:
                    raise nncf.ValidationError(
                        "Invalid node order parsed from graph line - "
                        "must specify either `node_id` or a pair of `edge_start_id`/`edge_end_id`!"
                    )
            else:
                if edge_start_id is None or edge_end_id is None:
                    raise nncf.ValidationError(
                        "Invalid node order - must specify both `edge_start_id` and `edge_end_id` "
                        "if node_id is None!"
                    )
            self.node_id = node_id
            self.edge_start_id = edge_start_id
            self.edge_end_id = edge_end_id

        def __lt__(self, other: "LineOrder"):
            if self.node_id is not None:
                if other.node_id is None:
                    return True
                if self.node_id < other.node_id:
                    return True
                return False
            if other.node_id is not None:
                return False
            if self.edge_start_id < other.edge_start_id:
                return True
            if self.edge_start_id > other.edge_start_id:
                return False
            if self.edge_end_id < other.edge_end_id:
                return True
            return False

    def graph_key(line: str) -> LineOrder:
        extract_ids_regex = r'^"(\d+) '
        start_id_matches = re.search(extract_ids_regex, line)
        if start_id_matches is None:
            raise nncf.InternalError(f"Could not parse first node ID in node name: {line}")
        start_id = int(start_id_matches.group(1))
        edge_indicator = " -> "
        if edge_indicator in line:
            end_node_and_attrs_str = line.split(edge_indicator)[1]
            end_id_matches = re.search(extract_ids_regex, end_node_and_attrs_str)
            if end_id_matches is None:
                raise nncf.InternalError(f"Could not parse end node ID in node name: {end_node_and_attrs_str}")
            end_id = int(end_id_matches.group(1))
            return LineOrder(edge_start_id=start_id, edge_end_id=end_id)
        return LineOrder(node_id=int(start_id))

    sorted_content = sorted(content, key=graph_key)
    with open(path, "w", encoding="utf8") as f:
        f.write(start_line)
        f.writelines(sorted_content)
        f.write(end_line)


def _build_node_id_vs_attrs_dict(
    nx_graph: nx.DiGraph, id_from_attr: bool = False
) -> Dict[Union[int, str], Dict[str, str]]:
    retval: Dict[Union[int, str], Dict[str, str]] = {}
    for node_name, node_attrs in nx_graph.nodes.items():
        # When read a dot graph dumped by pydot the extra '\n' symbol appears as a graph node.
        # https://github.com/networkx/networkx/issues/5686
        if node_name == "\\n":  # bug - networkx/networkx#5686
            continue
        if id_from_attr:
            node_identifier = int(node_attrs["id"])
        else:
            node_identifier = node_name
        retval[node_identifier] = {k: str(v).strip('"') for k, v in node_attrs.items()}
    return retval


def _build_edge_vs_attrs_dict(
    nx_graph: nx.DiGraph, id_from_attr: bool = False
) -> Dict[Tuple[Union[int, str], Union[int, str]], Dict[str, str]]:
    retval = {}
    for edge_tuple, edge_attrs in nx_graph.edges.items():
        from_node_name, to_node_name = edge_tuple
        if id_from_attr:
            from_node, to_node = nx_graph.nodes[from_node_name], nx_graph.nodes[to_node_name]
            edge_id = int(from_node["id"]), int(to_node["id"])
        else:
            edge_id = from_node_name, to_node_name
        retval[edge_id] = {k: str(v).strip('"') for k, v in edge_attrs.items()}
    return retval


def check_nx_graph(
    nx_graph: nx.DiGraph, expected_graph: nx.DiGraph, check_edge_attrs: bool = False, unstable_node_names: bool = False
) -> None:
    id_vs_attrs = _build_node_id_vs_attrs_dict(nx_graph, id_from_attr=unstable_node_names is True)
    expected_id_vs_attrs = _build_node_id_vs_attrs_dict(expected_graph, id_from_attr=unstable_node_names is True)

    for node_identifier, expected_attrs in expected_id_vs_attrs.items():
        assert node_identifier in id_vs_attrs, f"Expected to find node {node_identifier}, but there is no such node."
        expected_attrs = dict(sorted(expected_attrs.items()))
        attrs = dict(sorted(id_vs_attrs[node_identifier].items()))
        assert (
            expected_attrs == attrs
        ), f"Incorrect attributes for node {node_identifier}. Expected {expected_attrs}, but actual {attrs}."

    edge_vs_attrs = _build_edge_vs_attrs_dict(nx_graph, id_from_attr=unstable_node_names is True)
    expected_edge_vs_attrs = _build_edge_vs_attrs_dict(nx_graph, id_from_attr=unstable_node_names is True)
    assert edge_vs_attrs.keys() == expected_edge_vs_attrs.keys()

    if check_edge_attrs:
        for expected_edge_tuple, expected_attrs in expected_edge_vs_attrs.items():
            expected_attrs = dict(sorted(expected_attrs.items()))
            attrs = dict(sorted(edge_vs_attrs[expected_edge_tuple].items()))
            assert attrs == expected_attrs, (
                f"Incorrect edge attributes for edge {expected_edge_tuple}."
                f" expected {expected_attrs}, but actual {attrs}."
            )


def compare_nx_graph_with_reference(
    nx_graph: nx.DiGraph,
    path_to_dot: str,
    sort_dot_graph: bool = True,
    check_edge_attrs: bool = False,
    unstable_node_names: bool = False,
) -> None:
    """
    Checks whether the two nx.DiGraph are identical. The first one is 'nx_graph' argument
    and the second graph is read from the absolute path - 'path_to_dot'.
    Also, could dump the graph, based in the global variable NNCF_TEST_REGEN_DOT.
    If 'sort_dot_graph' is True sorts the second graph before dumping.
    If 'check_edge_attrs' is True checks edge attributes of the graphs.
    :param nx_graph: The first nx.DiGraph.
    :param path_to_dot: The absolute path to the second nx.DiGraph.
    :param sort_dot_graph: whether to call sort_dot() function on the second graph.
    :param check_edge_attrs: whether to check edge attributes of the graphs.
    :return: None
    """
    dot_dir = Path(path_to_dot).parent
    # validate .dot file manually!
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(dot_dir):
            os.makedirs(dot_dir)
        write_dot_graph(nx_graph, Path(path_to_dot))
        if sort_dot_graph:
            sort_dot(path_to_dot)

    expected_graph = nx.DiGraph(read_dot_graph(Path(path_to_dot)))
    check_nx_graph(nx_graph, expected_graph, check_edge_attrs, unstable_node_names)
