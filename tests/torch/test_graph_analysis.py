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
from collections import Counter

import networkx as nx

from nncf.common.graph import NNCFGraphEdge
from nncf.common.graph import NNCFGraphPatternIO
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import Dtype
from tests.common.quantization.mock_graphs import get_mock_nncf_node_attrs
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import mark_input_ports_lexicographically_based_on_input_node_key


def test_graph_pattern_io_building():
    mock_graph = nx.DiGraph()
    #   A
    # /   \
    # B   |
    # |   |
    # C   |
    # \   /
    #   D
    # / | \
    # E F G
    # |
    # H

    node_keys = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for node_key in node_keys:
        mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
        mock_graph.add_node(node_key, **mock_node_attrs)

    mock_graph.add_edges_from(
        [("A", "B"), ("A", "D"), ("B", "C"), ("C", "D"), ("D", "E"), ("D", "F"), ("D", "G"), ("E", "H")]
    )
    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)

    graph = get_nncf_graph_from_mock_nx_graph(mock_graph)

    def make_mock_edge(
        from_node_name: NNCFNodeName, to_node_name: NNCFNodeName, input_port_id: int, output_port_id: int
    ):
        return NNCFGraphEdge(
            get_node(from_node_name),
            get_node(to_node_name),
            input_port_id=input_port_id,
            output_port_id=output_port_id,
            tensor_shape=[1, 1, 1, 1],
            dtype=Dtype.FLOAT,
            parallel_input_port_ids=[],
        )

    def get_node(name: NNCFNodeName):
        return graph.get_node_by_name(name)

    ref_patterns_and_ios = [
        (
            ["/A_0", "/B_0"],
            NNCFGraphPatternIO(
                input_edges=[],
                output_edges=[
                    make_mock_edge("/B_0", "/C_0", input_port_id=0, output_port_id=0),
                    make_mock_edge("/A_0", "/D_0", input_port_id=0, output_port_id=1),
                ],
            ),
        ),
        (
            ["/C_0"],
            NNCFGraphPatternIO(
                input_edges=[make_mock_edge("/B_0", "/C_0", input_port_id=0, output_port_id=0)],
                output_edges=[make_mock_edge("/C_0", "/D_0", input_port_id=1, output_port_id=0)],
            ),
        ),
        (
            ["/A_0", "/B_0", "/C_0"],
            NNCFGraphPatternIO(
                input_edges=[],
                output_edges=[
                    make_mock_edge("/C_0", "/D_0", input_port_id=1, output_port_id=0),
                    make_mock_edge("/A_0", "/D_0", input_port_id=0, output_port_id=1),
                ],
            ),
        ),
        (
            ["/D_0"],
            NNCFGraphPatternIO(
                input_edges=[
                    make_mock_edge("/C_0", "/D_0", input_port_id=1, output_port_id=0),
                    make_mock_edge("/A_0", "/D_0", input_port_id=0, output_port_id=1),
                ],
                output_edges=[
                    make_mock_edge("/D_0", "/E_0", input_port_id=0, output_port_id=0),
                    make_mock_edge("/D_0", "/F_0", input_port_id=0, output_port_id=1),
                    make_mock_edge("/D_0", "/G_0", input_port_id=0, output_port_id=2),
                ],
            ),
        ),
        (
            ["/E_0", "/F_0", "/H_0"],
            NNCFGraphPatternIO(
                input_edges=[
                    make_mock_edge("/D_0", "/E_0", input_port_id=0, output_port_id=0),
                    make_mock_edge("/D_0", "/F_0", input_port_id=0, output_port_id=1),
                ],
                output_edges=[],
            ),
        ),
        (
            ["/G_0"],
            NNCFGraphPatternIO(
                input_edges=[make_mock_edge("/D_0", "/G_0", input_port_id=0, output_port_id=2)], output_edges=[]
            ),
        ),
    ]

    for node_names_list, ref_pattern_io in ref_patterns_and_ios:
        node_key_pattern = [graph.get_node_key_by_id(graph.get_node_by_name(name).node_id) for name in node_names_list]
        test_pattern_io = graph.get_nncf_graph_pattern_io(node_key_pattern)
        assert Counter(test_pattern_io.input_edges) == Counter(ref_pattern_io.input_edges)
        assert Counter(test_pattern_io.output_edges) == Counter(ref_pattern_io.output_edges)
