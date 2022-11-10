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
