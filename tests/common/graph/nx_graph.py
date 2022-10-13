"""
 Copyright (c) 2022 Intel Corporation
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

import os
from functools import partial
from pathlib import Path
import networkx as nx

from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph


def sort_dot(path):
    with open(path, 'r', encoding='utf8') as f:
        content = f.readlines()
    start_line = 'strict digraph  {\n'
    end_line = '}\n'
    content.remove(start_line)
    content.remove(end_line)

    def graph_key(line, offset):
        key = line.split(' ')[0].replace('"', '')
        if '->' in line:
            key += line.split(' ')[3].replace('"', '')
            return int(key) + offset
        return int(key)

    sorted_content = sorted(content, key=partial(graph_key, offset=len(content)))
    with open(path, 'w', encoding='utf8') as f:
        f.write(start_line)
        f.writelines(sorted_content)
        f.write(end_line)


def check_nx_graph(nx_graph: nx.DiGraph, expected_graph: nx.DiGraph, check_edge_attrs: bool = False) -> None:
    # Check nodes attrs
    for node_name, node_attrs in nx_graph.nodes.items():
        expected_attrs = {k: str(v).strip('"') for k, v in expected_graph.nodes[node_name].items()}
        attrs = {k: str(v) for k, v in node_attrs.items()}
        assert expected_attrs == attrs

    assert nx.DiGraph(expected_graph).edges == nx_graph.edges

    if check_edge_attrs:
        for nx_graph_edges, expected_graph_edges in zip(nx_graph.edges.data(), expected_graph.edges.data()):
            for nx_edge_attrs, expected_graph_edge_attrs in zip(nx_graph_edges, expected_graph_edges):
                if isinstance(nx_edge_attrs, dict):
                    nx_edge_attrs['label'] = str(nx_edge_attrs['label'])
                    if not isinstance(expected_graph_edge_attrs['label'], list):
                        expected_graph_edge_attrs['label'] = expected_graph_edge_attrs['label'].replace('"', '')
                    else:
                        expected_graph_edge_attrs['label'] = str(expected_graph_edge_attrs['label'])
                assert nx_edge_attrs == expected_graph_edge_attrs


def compare_nx_graph_with_reference(nx_graph: nx.DiGraph, path_to_dot: str,
                                    sort_dot_graph=True, check_edge_attrs: bool = False) -> None:
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
        write_dot_graph(nx_graph, path_to_dot)
        if sort_dot_graph:
            sort_dot(path_to_dot)

    expected_graph = read_dot_graph(path_to_dot)
    check_nx_graph(nx_graph, expected_graph, check_edge_attrs)
