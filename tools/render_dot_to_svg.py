"""
Useful when the .dot file is so large that no tool, online or offline, can effectively visualize it.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from nncf.common.utils.dot_file_rw import read_dot_graph


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Input .dot file", required=True)
    args = parser.parse_args(args=argv)

    graph = nx.DiGraph(read_dot_graph(args.input_file))
    A = to_agraph(graph)
    A.layout("dot")
    png_path = Path(args.input_file).with_suffix(".svg")
    A.draw(str(png_path))


if __name__ == "__main__":
    main(sys.argv[1:])
