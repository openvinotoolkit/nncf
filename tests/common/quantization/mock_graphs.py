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

import random
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import networkx as nx

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from tests.common.quantization.metatypes import METATYPES_FOR_TEST
from tests.common.quantization.metatypes import TestMetatype

OP_NAMES_IN_TEST_WITH_MODULE_ATTRIBUTES = [
    "conv1d",
    "conv2d",
    "conv3d",
    "linear",
    "batch_norm",
    "group_norm",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "embedding",
    "embedding_bag",
]


class NodeWithType:
    def __init__(
        self,
        name: str,
        op_metatype: TestMetatype,
        op_type: str = None,
        layer_attributes: Optional[BaseLayerAttributes] = None,
    ):
        self.node_name = name
        self.node_op_metatype = op_metatype
        self.node_op_type = op_type
        self.layer_attributes = layer_attributes


def create_mock_graph(
    nodes: List[NodeWithType], node_edges: List[Tuple[str, str]], edges_attrs: Optional[Tuple[Any]] = None
) -> nx.DiGraph:
    mock_graph = nx.DiGraph()
    for node in nodes:
        mock_node_attrs = get_mock_nncf_node_attrs(
            op_name=node.node_name,
            metatype=node.node_op_metatype,
            type_=node.node_op_type,
            layer_attributes=node.layer_attributes,
        )
        mock_graph.add_node(node.node_name, **mock_node_attrs)
    if edges_attrs:
        for (edge_from, edge_to), attr in zip(node_edges, edges_attrs):
            mock_graph.add_edge(edge_from, edge_to, **attr)
    else:
        mock_graph.add_edges_from(node_edges)
    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return mock_graph


def mark_input_ports_lexicographically_based_on_input_node_key(graph: nx.DiGraph):
    for node_key in graph.nodes:
        input_edges = graph.in_edges(node_key)
        sorted_input_edges = sorted(input_edges, key=lambda x: x[0])
        for idx, edge in enumerate(sorted_input_edges):
            graph.edges[edge][NNCFGraph.INPUT_PORT_ID_EDGE_ATTR] = idx


def get_nncf_graph_from_mock_nx_graph(nx_graph: nx.DiGraph, nncf_graph_cls=NNCFGraph) -> NNCFGraph:
    mock_graph = nncf_graph_cls()
    key_vs_id = {}
    edge_vs_output_idx_and_creator_id: Dict[Tuple[str, str], Tuple[int, int]] = {}
    from networkx.algorithms.dag import lexicographical_topological_sort

    for idx, curr_node_key in enumerate(lexicographical_topological_sort(nx_graph)):
        node = nx_graph.nodes[curr_node_key]
        if NNCFNode.NODE_NAME_ATTR in node:
            node_name = node[NNCFNode.NODE_NAME_ATTR]
        else:
            node_name = "/" + curr_node_key + "_0"

        node_type = node.get(NNCFNode.NODE_TYPE_ATTR, curr_node_key)
        layer_attributes = node.get(NNCFNode.LAYER_ATTRIBUTES)

        if NNCFNode.METATYPE_ATTR in node:
            metatype = node[NNCFNode.METATYPE_ATTR]
        else:
            metatype = METATYPES_FOR_TEST.get_operator_metatype_by_op_name(node_type)
            if metatype is not UnknownMetatype and metatype.get_subtypes():
                subtype = metatype.determine_subtype(layer_attributes=layer_attributes)
                if subtype is not None:
                    metatype = subtype

        node_id = idx
        node = mock_graph.add_nncf_node(
            node_name=node_name,
            node_type=node_type,
            node_metatype=metatype,
            layer_attributes=layer_attributes,
            node_id_override=idx,
        )
        key_vs_id[curr_node_key] = node_id

        preds = list(nx_graph.predecessors(curr_node_key))
        for pred_idx, pred in enumerate(preds):
            in_edge = (pred, curr_node_key)
            out_idx, creator_id = edge_vs_output_idx_and_creator_id[in_edge]
            edge_data = nx_graph.edges[in_edge]
            dtype = edge_data.get(NNCFGraph.DTYPE_EDGE_ATTR, Dtype.FLOAT)
            shape = edge_data.get(NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR, [1, 1, 1, 1])
            mock_graph.add_edge_between_nncf_nodes(
                creator_id, node_id, shape, input_port_id=pred_idx, output_port_id=out_idx, dtype=dtype
            )

        for out_idx, out_edge in enumerate(nx_graph.out_edges(curr_node_key)):
            edge_vs_output_idx_and_creator_id[out_edge] = (out_idx, node.node_id)
    return mock_graph


def get_two_branch_mock_model_graph() -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

    #   (0 /A)
    #      |
    #   (1 /B)
    #   /     \
    # (2 /C) (3 /D)
    #  |       |
    # (4 /E)   |
    #   \     /
    #   (5 /F)
    #     |
    #   (6 /G)
    #     |
    #   (7 /H)

    node_keys = ["A", "B", "C", "D", "E", "F", "G", "H"]

    for node_key in node_keys:
        mock_nx_graph.add_node(node_key)

    mock_nx_graph.add_edges_from(
        [("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("E", "F"), ("D", "F"), ("F", "G"), ("G", "H")]
    )

    mark_input_ports_lexicographically_based_on_input_node_key(mock_nx_graph)
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


MOCK_OPERATOR_NAME = "conv_transpose2d"


def get_mock_nncf_node_attrs(op_name=None, scope_str=None, metatype=None, type_=None, layer_attributes=None):
    op_name_to_set = op_name if op_name is not None else MOCK_OPERATOR_NAME
    if type_ is None:
        type_ = op_name_to_set
    if scope_str is None:
        scope_str = ""
    output = {
        NNCFNode.NODE_NAME_ATTR: f"{scope_str}/{op_name_to_set}_0",
        NNCFNode.NODE_TYPE_ATTR: type_,
    }
    for attr_name, attr_val in [(NNCFNode.METATYPE_ATTR, metatype), (NNCFNode.LAYER_ATTRIBUTES, layer_attributes)]:
        if attr_val is not None:
            output[attr_name] = attr_val

    return output


def _add_nodes_with_layer_attrs(
    nx_graph: nx.DiGraph,
    node_keys: List[str],
    layer_attrs: Dict[str, BaseLayerAttributes],
    metatypes: Dict[str, OperatorMetatype] = None,
) -> nx.DiGraph:
    for node_key in node_keys:
        metatype = None
        if metatypes is not None and node_key in metatypes:
            metatype = metatypes[node_key]
        nx_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key, metatype=metatype))

        if node_key in layer_attrs:
            nx_graph.nodes[node_key][NNCFNode.LAYER_ATTRIBUTES] = layer_attrs[node_key]

    return nx_graph


def get_mock_model_graph_with_mergeable_pattern(
    conv2d_metatype=None, batchnorm_metatype=None, relu_metatype=None
) -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)
    #    |
    # (batch_norm)
    #    |
    #  (RELU)
    #    |
    #   (B)

    node_keys = ["conv2d", "batch_norm", "relu", "A", "B"]

    layer_attrs = {
        "conv2d": ConvolutionLayerAttributes(
            weight_requires_grad=False,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=1,
            transpose=False,
            padding_values=[0, 0, 0, 0],
        )
    }
    metatypes = {
        "conv2d": conv2d_metatype,
        "batch_norm": batchnorm_metatype,
        "relu": relu_metatype,
    }
    mock_nx_graph = _add_nodes_with_layer_attrs(mock_nx_graph, node_keys, layer_attrs, metatypes)

    mock_nx_graph.add_edges_from(
        [
            ("A", "conv2d", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("conv2d", "batch_norm", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("batch_norm", "relu", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("relu", "B", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
        ]
    )
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


def get_mock_model_graph_with_no_mergeable_pattern(
    conv2d_metatype=None, batchnorm_metatype=None, relu_metatype=None
) -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)
    #    |
    #   (C)
    #    |
    # (batch_norm)
    #    |
    #   (D)
    #    |
    #  (relu)
    #    |
    #   (B)

    node_keys = ["conv2d", "batch_norm", "relu", "A", "B", "C", "D"]

    layer_attrs = {
        "conv2d": ConvolutionLayerAttributes(
            weight_requires_grad=False,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=1,
            transpose=False,
            padding_values=[0, 0, 0, 0],
        )
    }
    metatypes = {
        "conv2d": conv2d_metatype,
        "batch_norm": batchnorm_metatype,
        "relu": relu_metatype,
    }
    mock_nx_graph = _add_nodes_with_layer_attrs(mock_nx_graph, node_keys, layer_attrs, metatypes)

    mock_nx_graph.add_edges_from(
        [
            ("A", "conv2d", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("conv2d", "C", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("C", "batch_norm", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("batch_norm", "D", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("D", "relu", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("relu", "B", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
        ]
    )
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


def get_mock_model_graph_with_broken_output_edge_pattern(
    conv2d_metatype=None, batchnorm_metatype=None, relu_metatype=None
) -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)----\
    #    |         |
    # (batch_norm) |
    #    |         |
    #  (RELU)      |
    #    |         |
    #   (C)--------/
    #    |
    #   (B)

    node_keys = ["conv2d", "batch_norm", "relu", "A", "B", "C"]
    layer_attrs = {
        "conv2d": ConvolutionLayerAttributes(
            weight_requires_grad=False,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=1,
            transpose=False,
            padding_values=[0, 0, 0, 0],
        )
    }
    metatypes = {
        "conv2d": conv2d_metatype,
        "batch_norm": batchnorm_metatype,
        "relu": relu_metatype,
    }
    mock_nx_graph = _add_nodes_with_layer_attrs(mock_nx_graph, node_keys, layer_attrs, metatypes)

    mock_nx_graph.add_edges_from(
        [
            ("A", "conv2d", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("conv2d", "batch_norm", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("conv2d", "C", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 1}),
            ("batch_norm", "relu", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("relu", "C", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
            ("C", "B", {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 0}),
        ]
    )
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


def get_ip_graph_for_test(nncf_graph: NNCFGraph) -> InsertionPointGraph:
    pre_hooks = []
    post_hooks = []
    for node in nncf_graph.get_all_nodes():
        in_edges = nncf_graph.get_input_edges(node)
        for in_edge in in_edges:
            ip = PreHookInsertionPoint(node.node_name, in_edge.input_port_id)
            pre_hooks.append(ip)

        # TODO (vshampor): remove
        # if issubclass(node.metatype, PTSplitMetatype):
        #     continue
        ip = PostHookInsertionPoint(node.node_name)
        post_hooks.append(ip)

    ip_graph = InsertionPointGraph(
        nncf_graph,
        allowed_pre_hook_insertion_points=pre_hooks,
        allowed_post_hook_insertion_points=post_hooks,
    )
    return ip_graph


def get_node_name(op_name: str, call_order: int = 0) -> str:
    return f"/{op_name}_{call_order}"


def get_randomly_connected_model_graph(op_name_keys: Set[str]) -> nx.DiGraph:
    graph_len = len(op_name_keys)
    mock_graph = nx.generators.gnc_graph(graph_len, None, 0)

    shuffled_op_names = random.sample(sorted(op_name_keys), len(op_name_keys))
    for idx, (_, node) in enumerate(mock_graph.nodes.items()):
        op_name = shuffled_op_names[idx]
        node[NNCFNode.NODE_NAME_ATTR] = get_node_name(shuffled_op_names[idx])
        node[NNCFNode.NODE_TYPE_ATTR] = op_name
        if op_name in OP_NAMES_IN_TEST_WITH_MODULE_ATTRIBUTES:
            node[NNCFNode.LAYER_ATTRIBUTES] = MagicMock()
    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return mock_graph


def get_sequentially_connected_model_graph(op_name_keys: List[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    node_key_appearances = {k: 0 for k in op_name_keys}

    actual_keys = []
    for node_key in op_name_keys:
        attrs = {
            NNCFNode.NODE_NAME_ATTR: get_node_name(node_key, call_order=node_key_appearances[node_key]),
            NNCFNode.NODE_TYPE_ATTR: node_key,
        }

        if node_key in OP_NAMES_IN_TEST_WITH_MODULE_ATTRIBUTES:
            attrs[NNCFNode.LAYER_ATTRIBUTES] = MagicMock()
        actual_key = node_key + "_{}".format(node_key_appearances[node_key])
        graph.add_node(actual_key, **attrs)
        node_key_appearances[node_key] += 1
        actual_keys.append(actual_key)

    edges = [(actual_keys[i], actual_keys[i + 1]) for i in range(0, len(actual_keys) - 1)]
    for from_key, to_key in edges:
        graph.add_edge(from_key, to_key)

    mark_input_ports_lexicographically_based_on_input_node_key(graph)
    return graph
