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
import copy
import pathlib
from collections import defaultdict
from typing import Dict

import networkx as nx  # type: ignore


def write_dot_graph(G: nx.DiGraph, path: pathlib.Path) -> None:
    # NOTE: writing dot files with colons even in labels or other node/edge/graph attributes leads to an
    # error. See https://github.com/networkx/networkx/issues/5962. If `relabel` is True in this function,
    # then the colons (:) will be replaced with (^) symbols.
    relabeled = relabel_graph_for_dot_visualization(G)
    nx.nx_pydot.write_dot(relabeled, str(path))


def get_graph_without_data(G: nx.DiGraph) -> nx.DiGraph:
    """
    Returns the data-less version of the given nx.Digraph. The new graph has no data in nodes and edges.

    :return: nx.DiGraph without data in nodes and edges
    """
    out_graph = nx.DiGraph()
    for node_key in G.nodes(data=False):
        out_graph.add_node(node_key)
    for u, v in G.edges(data=False):
        out_graph.add_edge(u, v)
    return out_graph


def read_dot_graph(path: pathlib.Path) -> nx.DiGraph:
    loaded = nx.DiGraph(nx.nx_pydot.read_dot(str(path)))
    return relabel_graph_for_dot_visualization(loaded, from_reference=True)


RESERVED_CHAR = ":"
REPLACEMENT_CHAR = "^"


def _maybe_escape_colons_in_attrs(data: Dict[str, str]) -> None:
    for attr_name in data:
        attr_val = str(data[attr_name])
        if RESERVED_CHAR in attr_val and not (attr_val[0] == '"' or attr_val[-1] == '"'):
            data[attr_name] = '"' + data[attr_name] + '"'  # escaped colons are allowed


def _unescape_colons_in_attrs_with_colons(data: Dict[str, str]) -> None:
    for attr_name in data:
        attr_val = data[attr_name]
        if RESERVED_CHAR in attr_val and (attr_val[0] == '"' and attr_val[-1] == '"'):
            data[attr_name] = data[attr_name][1:-1]


def _remove_cosmetic_labels(graph: nx.DiGraph) -> None:
    for node_name, node_data in graph.nodes(data=True):
        if "label" in node_data:
            label = node_data["label"]
            if node_name == label or '"' + node_name + '"' == label:
                del node_data["label"]


def _add_cosmetic_labels(graph: nx.DiGraph, relabeled_node_mapping: Dict[str, str]) -> None:
    for original_name, dot_name in relabeled_node_mapping.items():
        node_data = graph.nodes[dot_name]
        if "label" not in node_data:
            node_data["label"] = '"' + original_name + '"'


def relabel_graph_for_dot_visualization(nx_graph: nx.Graph, from_reference: bool = False) -> nx.DiGraph:
    """
    Relabels NetworkX graph nodes to exclude reserved symbols in keys.
        In case replaced names match for two different nodes, integer index is added to its keys.
        While nodes keys are being updated, visualized nodes names corresponds to the original nodes names.

    :param nx_graph: NetworkX graph to visualize via dot.
    :return: NetworkX graph with reserved symbols in nodes keys replaced.
    """

    nx_graph = copy.deepcopy(nx_graph)

    # .dot format reserves ':' character in node names
    if not from_reference:
        # dumping to disk
        __CHARACTER_REPLACE_FROM = RESERVED_CHAR
        __CHARACTER_REPLACE_TO = REPLACEMENT_CHAR
    else:
        # loading from disk
        __CHARACTER_REPLACE_FROM = REPLACEMENT_CHAR
        __CHARACTER_REPLACE_TO = RESERVED_CHAR

    hits: Dict[str, int] = defaultdict(lambda: 0)
    mapping = {}
    for original_name in nx_graph.nodes():
        if not isinstance(original_name, str):
            continue
        dot_name = original_name.replace(__CHARACTER_REPLACE_FROM, __CHARACTER_REPLACE_TO)
        hits[dot_name] += 1
        if hits[dot_name] > 1:
            dot_name = f"{dot_name}_{hits}"
        if original_name != dot_name:
            mapping[original_name] = dot_name

    relabeled_graph = nx.relabel_nodes(nx_graph, mapping)

    if not from_reference:
        # dumping to disk
        _add_cosmetic_labels(relabeled_graph, mapping)
        for _, node_data in relabeled_graph.nodes(data=True):
            _maybe_escape_colons_in_attrs(node_data)
        for _, _, edge_data in relabeled_graph.edges(data=True):
            _maybe_escape_colons_in_attrs(edge_data)
    else:
        # loading from disk
        _remove_cosmetic_labels(relabeled_graph)
        for _, node_data in relabeled_graph.nodes(data=True):
            _unescape_colons_in_attrs_with_colons(node_data)
        for _, _, edge_data in relabeled_graph.edges(data=True):
            _unescape_colons_in_attrs_with_colons(edge_data)

    return relabeled_graph
