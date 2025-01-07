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
