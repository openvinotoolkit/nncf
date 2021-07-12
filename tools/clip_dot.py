#pylint:skip-file
import sys
from argparse import ArgumentParser

import networkx as nx


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', help='Input .dot file',
                        required=True)
    parser.add_argument('-s', '--start_id', help='Start ID (inclusive)',
                        required=True)
    parser.add_argument('-f', '--finish_id', help='Finish ID (inclusive)', required=True)
    parser.add_argument('-o', '--output_file', help='Output .dot file', required=True)
    args = parser.parse_args(args=argv)

    graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(args.input_file))

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
        raise RuntimeError("Could not find the node with ID {} to start from!".format(args.start_id))

    for edge in nx.edge_bfs(graph, start_key, orientation='ignore'):
        from_key, to_key, _ = edge
        id_portion = from_key.split()[0]
        has_id = id_portion.isdigit()
        end_key = from_key
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

    nx.drawing.nx_pydot.write_dot(new_graph, args.output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
