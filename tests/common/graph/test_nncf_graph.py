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

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.patterns import GraphPattern


def test_find_matching_subgraphs():
    nncf_graph = NNCFGraph()
    nodes = []
    for node_id in "abcdef":
        nodes.append(nncf_graph.add_nncf_node(node_id, node_id, f"metatype_{node_id}"))

    for i in range(1, len(nodes)):
        nncf_graph.add_edge_between_nncf_nodes(
            from_node_id=nodes[i - 1].node_id,
            to_node_id=nodes[i].node_id,
            tensor_shape=[1],
            input_port_id=0,
            output_port_id=0,
            dtype=Dtype.FLOAT,
        )

    graph_pattern = GraphPattern()
    for patterns in ["ab", "def"]:
        graph_part = GraphPattern()
        pattern_nodes = []
        for metatype in patterns:
            pattern_nodes.append(graph_part.add_node(**{GraphPattern.METATYPE_ATTR: metatype}))
        for i in range(1, len(pattern_nodes)):
            graph_part.add_edge(pattern_nodes[i - 1], pattern_nodes[i])
        graph_pattern.add_pattern_alternative(graph_part)

    matches = nncf_graph.find_matching_subgraphs(graph_pattern)
    assert len(matches) == 2
    for match in matches:
        if len(match) == 3:
            assert match == nodes[3:]
            continue
        assert len(match) == 2
        assert match == nodes[:2]
