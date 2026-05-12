# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFGraphEdge
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


def _get_default_nncf_graph_edge(from_node, to_node, input_port_id, output_port_id):
    return NNCFGraphEdge(
        from_node,
        to_node,
        input_port_id=input_port_id,
        output_port_id=output_port_id,
        tensor_shape=(1, 2, 3),
        dtype="dummy",
    )


def test_parallel_edges():
    nncf_graph = NNCFGraph()
    nodes = []
    for node in "abc":
        nodes.append(nncf_graph.add_nncf_node(node, f"type_{node}", f"metatype_{node}"))

    for input_port_id in range(5):
        nncf_graph.add_edge_between_nncf_nodes(
            nodes[0].node_id,
            nodes[1].node_id,
            input_port_id=input_port_id,
            output_port_id=0,
            tensor_shape=(1, 2, 3),
            dtype="dummy",
        )
    nncf_graph.add_edge_between_nncf_nodes(
        nodes[0].node_id,
        nodes[2].node_id,
        input_port_id=10,
        output_port_id=15,
        tensor_shape=(1, 2, 3),
        dtype="dummy",
    )
    output_edges = nncf_graph.get_output_edges(nodes[0])
    input_edges = nncf_graph.get_input_edges(nodes[1])
    assert len(input_edges) == 5
    assert len(output_edges) == 6
    assert input_edges == output_edges[:-1]
    for input_port_id, edge in enumerate(input_edges):
        ref_edge = _get_default_nncf_graph_edge(
            nodes[0],
            nodes[1],
            input_port_id=input_port_id,
            output_port_id=0,
        )
        assert ref_edge == edge

    ordinary_edge = _get_default_nncf_graph_edge(
        nodes[0],
        nodes[2],
        input_port_id=10,
        output_port_id=15,
    )
    assert ordinary_edge == output_edges[-1]


def test_raise_error_for_duplicated_edge():
    nncf_graph = NNCFGraph()
    nodes = []
    for node in "abc":
        nodes.append(nncf_graph.add_nncf_node(node, f"type_{node}", f"metatype_{node}"))

    # First edge from port 1 to port 1 - OK
    nncf_graph.add_edge_between_nncf_nodes(
        nodes[0].node_id,
        nodes[1].node_id,
        input_port_id=1,
        output_port_id=1,
        tensor_shape=(1, 2, 3),
        dtype="dummy",
    )
    # Second edge from port 1 to port 1 - ValueError
    with pytest.raises(ValueError):
        nncf_graph.add_edge_between_nncf_nodes(
            nodes[0].node_id,
            nodes[1].node_id,
            input_port_id=1,
            output_port_id=1,
            tensor_shape=(1, 2, 3),
            dtype="dummy",
        )


def test_multi_edges():
    nncf_graph = NNCFGraph()
    nodes = []
    for node in "ab":
        nodes.append(nncf_graph.add_nncf_node(node, f"type_{node}", f"metatype_{node}"))

    for port_id in range(5):
        nncf_graph.add_edge_between_nncf_nodes(
            nodes[0].node_id,
            nodes[1].node_id,
            input_port_id=port_id,
            output_port_id=port_id,
            tensor_shape=(1, 2, 3),
            dtype="dummy",
        )

    output_edges = nncf_graph.get_output_edges(nodes[0])
    input_edges = nncf_graph.get_input_edges(nodes[1])
    assert len(input_edges) == 5
    assert len(output_edges) == 5
    assert input_edges == output_edges
    for port_id, edge in enumerate(input_edges):
        ref_edge = _get_default_nncf_graph_edge(
            nodes[0],
            nodes[1],
            input_port_id=port_id,
            output_port_id=port_id,
        )
        assert ref_edge == edge


def test_remove_passthrough_node_with_single_input_and_multiple_outputs():
    nncf_graph = NNCFGraph()

    producer = nncf_graph.add_nncf_node("producer", "type_a", "metatype_a")
    target_node = nncf_graph.add_nncf_node("target_node", "type_b", "metatype_b")
    consumer_1 = nncf_graph.add_nncf_node("consumer_1", "type_c", "metatype_c")
    consumer_2 = nncf_graph.add_nncf_node("consumer_2", "type_d", "metatype_d")

    nncf_graph.add_edge_between_nncf_nodes(
        from_node_id=producer.node_id,
        to_node_id=target_node.node_id,
        tensor_shape=(1,),
        input_port_id=4,
        output_port_id=7,
        dtype=Dtype.FLOAT,
    )
    nncf_graph.add_edge_between_nncf_nodes(
        from_node_id=target_node.node_id,
        to_node_id=consumer_1.node_id,
        tensor_shape=(1,),
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    nncf_graph.add_edge_between_nncf_nodes(
        from_node_id=target_node.node_id,
        to_node_id=consumer_2.node_id,
        tensor_shape=(1,),
        input_port_id=1,
        output_port_id=1,
        dtype=Dtype.FLOAT,
    )

    nncf_graph.remove_passthrough_node(target_node)

    with pytest.raises(KeyError):
        nncf_graph.get_node_key_by_id(target_node.node_id)

    new_output_edges = nncf_graph.get_output_edges(producer)
    assert len(new_output_edges) == 2

    edge_to_consumer_1 = nncf_graph.get_edges(producer, consumer_1)[0]
    assert edge_to_consumer_1.input_port_id == 0
    assert edge_to_consumer_1.output_port_id == 7
    assert edge_to_consumer_1.tensor_shape == (1,)
    assert edge_to_consumer_1.dtype == Dtype.FLOAT

    edge_to_consumer_2 = nncf_graph.get_edges(producer, consumer_2)[0]
    assert edge_to_consumer_2.input_port_id == 1
    assert edge_to_consumer_2.output_port_id == 7
    assert edge_to_consumer_2.tensor_shape == (1,)
    assert edge_to_consumer_2.dtype == Dtype.FLOAT


def test_remove_passthrough_node_raises_if_more_than_one_input_edge():
    nncf_graph = NNCFGraph()

    producer_1 = nncf_graph.add_nncf_node("producer_1", "type_a", "metatype_a")
    producer_2 = nncf_graph.add_nncf_node("producer_2", "type_b", "metatype_b")
    target_node = nncf_graph.add_nncf_node("target_node", "type_c", "metatype_c")
    consumer = nncf_graph.add_nncf_node("consumer", "type_d", "metatype_d")

    nncf_graph.add_edge_between_nncf_nodes(
        from_node_id=producer_1.node_id,
        to_node_id=target_node.node_id,
        tensor_shape=(1,),
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    nncf_graph.add_edge_between_nncf_nodes(
        from_node_id=producer_2.node_id,
        to_node_id=target_node.node_id,
        tensor_shape=(1,),
        input_port_id=1,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    nncf_graph.add_edge_between_nncf_nodes(
        from_node_id=target_node.node_id,
        to_node_id=consumer.node_id,
        tensor_shape=(1,),
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )

    with pytest.raises(nncf.InternalError, match="Only one input edge is supported"):
        nncf_graph.remove_passthrough_node(target_node)
