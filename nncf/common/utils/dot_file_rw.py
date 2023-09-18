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

import pathlib

import networkx as nx


def write_dot_graph(G: nx.DiGraph, path: pathlib.Path):
    # NOTE: writing dot files with colons even in labels or other node/edge/graph attributes leads to an
    # error. See https://github.com/networkx/networkx/issues/5962. This limits the networkx version in
    # NNCF to 2.8.3 unless this is fixed upstream or an inconvenient workaround is made in NNCF.
    nx.nx_pydot.write_dot(G, str(path))


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
    return nx.nx_pydot.read_dot(str(path))
