"""
 Copyright (c) 2019-2020 Intel Corporation
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
from collections import Counter

from nncf.common.graph import NNCFGraphEdge
from nncf.common.graph import NNCFGraphPatternIO
from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import NoopMetatype


def test_graph_pattern_io_building():
    graph = PTNNCFGraph()
    #   1
    # /   \
    # 2   |
    # |   |
    # 3   |
    # \   /
    #   4
    # / | \
    # 5 6 7
    # |
    # 8

    node_keys = ['1', '2', '3', '4', '5', '6', '7', '8']
    for idx, node_key in enumerate(node_keys):
        graph.add_nncf_node(node_name=node_key,
                            node_type=node_key,
                            node_metatype=NoopMetatype,
                            node_id_override=idx + 1)

    id_defined_edges = [(1, 2), (1, 4), (2, 3), (3, 4), (4, 5),
                        (4, 6), (4, 7), (5, 8)]
    for edge in id_defined_edges:
        graph.add_edge_between_nncf_nodes(edge[0], edge[1], [1,], None, dtype=Dtype.FLOAT)

    def make_mock_edge(from_id: int, to_id: int):

        return NNCFGraphEdge(get_node(from_id),
                             get_node(to_id), [1,])

    def get_node(id_: int):
        return graph.get_node_by_id(id_)

    ref_patterns_and_ios = [
        (['1', '2'], NNCFGraphPatternIO(input_edges=[],
                                        output_edges=[make_mock_edge(2, 3),
                                                      make_mock_edge(1, 4)])),
        (['3'], NNCFGraphPatternIO(input_edges=[make_mock_edge(2, 3)],
                                   output_edges=[make_mock_edge(3, 4)])),
        (['1', '2', '3'], NNCFGraphPatternIO(input_edges=[],
                                             output_edges=[make_mock_edge(3, 4),
                                                           make_mock_edge(1, 4)])),
        (['4'], NNCFGraphPatternIO(input_edges=[make_mock_edge(3, 4),
                                                make_mock_edge(1, 4)],
                                   output_edges=[make_mock_edge(4, 5),
                                                 make_mock_edge(4, 6),
                                                 make_mock_edge(4, 7)])),
        (['5', '6', '8'], NNCFGraphPatternIO(input_edges=[make_mock_edge(4, 5),
                                                          make_mock_edge(4, 6)],
                                             output_edges=[])),
        (['7'], NNCFGraphPatternIO(input_edges=[make_mock_edge(4, 7)],
                                   output_edges=[]))
    ]

    for node_key_list, ref_pattern_io in ref_patterns_and_ios:
        pattern = [graph.get_node_key_by_id(graph.get_node_by_name(name).node_id) for name in node_key_list]
        test_pattern_io = graph.get_nncf_graph_pattern_io(pattern)
        assert Counter(test_pattern_io.input_edges) == Counter(ref_pattern_io.input_edges)
        assert Counter(test_pattern_io.output_edges) == Counter(ref_pattern_io.output_edges)
