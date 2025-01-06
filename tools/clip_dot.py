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


import sys
from argparse import ArgumentParser

import networkx as nx

import nncf
from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Input .dot file", required=True)
    parser.add_argument("-s", "--start_id", help="Start ID (inclusive)", required=True)
    parser.add_argument("-f", "--finish_id", help="Finish ID (inclusive)", required=True)
    parser.add_argument("-o", "--output_file", help="Output .dot file", required=True)
    args = parser.parse_args(args=argv)

    graph = nx.DiGraph(read_dot_graph(args.input_file))

    new_graph = nx.DiGraph()

    start_key = None
    for node_key in nx.lexicographical_topological_sort(graph):
        id_portion = node_key.split()[0]
        has_id = id_portion.isdigit()
        if has_id:
            curr_id = int(id_portion)
            if curr_id == int(args.start_id):
                start_key = node_key
                break

    if start_key is None:
        raise nncf.InternalError("Could not find the node with ID {} to start from!".format(args.start_id))

    for edge in nx.edge_bfs(graph, start_key, orientation="ignore"):
        from_key, to_key, _ = edge
        id_portion = from_key.split()[0]
        has_id = id_portion.isdigit()
        if has_id:
            curr_id = int(id_portion)
            if curr_id >= int(args.finish_id):
                break
        node_data = graph.nodes[from_key]
        new_graph.add_node(from_key, **node_data)
        edge_data = graph.edges[from_key, to_key]
        new_graph.add_edge(from_key, to_key, **edge_data)

    # for edge in nx.edge_bfs(graph, end_key, reverse=True):
    #     from_key, to_key = edge
    #     if from_key == start_key:
    #         break
    #     node_data = graph.nodes[from_key]
    #     new_graph.add_node(from_key, **node_data)
    #     edge_data = graph.edges[from_key, to_key]
    #     new_graph.add_edge(from_key, to_key, **edge_data)

    write_dot_graph(new_graph, args.output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
