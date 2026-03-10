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


from collections import defaultdict
from typing import Any, cast

import networkx as nx  # type: ignore
import torch
from torch import nn

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConstantLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.function_hook.graph.build_graph_mode import build_graph
from nncf.torch.function_hook.graph.graph_utils import ConstMeta
from nncf.torch.function_hook.graph.graph_utils import EdgeMeta
from nncf.torch.function_hook.graph.graph_utils import FunctionMeta
from nncf.torch.function_hook.graph.graph_utils import InOutMeta
from nncf.torch.function_hook.graph.graph_utils import NodeType
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.graph.graph import PTNNCFGraph


def get_node_type(type: NodeType, meta: ConstMeta | FunctionMeta | InOutMeta) -> str:
    """
    Convert a given NodeType to its corresponding string representation.

    :param type: The type of the node, which can be one of the following:
    :param meta: Metadata associated with the node.
    :return: The string representation of the node type.
    """
    if isinstance(meta, InOutMeta) and type == NodeType.input:
        return "nncf_model_input"
    if isinstance(meta, InOutMeta) and type == NodeType.output:
        return "nncf_model_output"
    if isinstance(meta, ConstMeta):
        return "nncf_model_const"
    if isinstance(meta, FunctionMeta):
        return meta.func_name
    msg = "Unexpected metadata type"
    raise nncf.InternalError(msg)


def get_name_of_node(meta: ConstMeta | FunctionMeta | InOutMeta) -> str:
    """
    Get the name of a node based on its metadata.

    :param meta: The metadata of the node.
    :return: The name of the node.
    """
    if isinstance(meta, ConstMeta):
        return meta.name_in_model
    if isinstance(meta, FunctionMeta):
        return meta.op_name
    if isinstance(meta, InOutMeta):
        return meta.name
    msg = "Unexpected metadata type"
    raise nncf.InternalError(msg)


def get_dtype(dtype: torch.dtype) -> Dtype:
    """
    Converts a torch dtype to a NNCF Dtype enum.

    :param dtype: The torch dtype to be converted.
    :return: The corresponding NNCF Dtype enum value.
    """
    if dtype in [torch.float, torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return Dtype.FLOAT
    return Dtype.INTEGER


def get_meta_type(node_type: str, meta: ConstMeta | FunctionMeta | InOutMeta) -> type[om.PTOperatorMetatype]:
    """
    Converts the node type and metadata into a PTOperatorMetatype object.
    :param node_type: The type of the node.
    :param meta: The metadata associated with the node.
    :return: The PTOperatorMetatype object.
    """
    node_metatype = cast(
        type[om.PTOperatorMetatype], om.PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
    )
    node_sub_meta_type: type[om.PTOperatorMetatype] | None = None
    if node_metatype.get_subtypes() and isinstance(meta, FunctionMeta):
        node_sub_meta_type = node_metatype.determine_subtype(function_args=meta.args, functions_kwargs=meta.kwargs)
    return node_sub_meta_type or node_metatype


def is_constant_input_node(nx_graph: nx.MultiDiGraph, node: int) -> bool:
    """
    Check if a node is a constant input node or constant subgraph:

    1) constant
    2) quantize_function -> constant

    :param nx_graph: The graph to check the node from.
    :param node: The node to check.
    :return: True if the node is a constant input node, False otherwise.
    """
    meta = nx_graph.nodes[node]["meta"]

    # 1) Input node is a constant node (parameter or buffer)
    if isinstance(meta, ConstMeta):
        return True

    # 2) Quantize node with constant input
    if (
        isinstance(meta, FunctionMeta)
        and meta.func_name in om.QUANTIZE_NODE_TYPES
        and isinstance(nx_graph.nodes[node]["meta"], FunctionMeta)
    ):
        return all(isinstance(nx_graph.nodes[s_node]["meta"], ConstMeta) for s_node, _ in nx_graph.in_edges(node))

    return False


def get_constant_port_ids(nx_graph: nx.MultiDiGraph, node: int) -> set[int]:
    """
    Get the indices of input ports corresponding to the constant node or subgraph.

    :param nx_graph: The graph to get the constant port IDs from.
    :param node: The node to get the constant port IDs from.
    :return: The list of input port indices with constants.
    """
    constant_port_ids: set[int] = set()

    for s_node, _, data in nx_graph.in_edges(node, data=True):
        if is_constant_input_node(nx_graph, s_node):
            meta = cast(EdgeMeta, data["meta"])
            constant_port_ids.add(meta.input_port)

    return constant_port_ids


def get_layer_attributes(
    nx_graph: nx.MultiDiGraph, node: int, meta: ConstMeta | FunctionMeta | InOutMeta
) -> BaseLayerAttributes | None:
    """
    Get the layer attributes of a node in the graph.

    :param nx_graph: The graph to get the layer attributes from.
    :param node: The node to get the layer attributes from.
    :param meta: The metadata associated with the node.
    :return: The layer attributes of the node.
    """
    if isinstance(meta, FunctionMeta):
        constant_port_ids = get_constant_port_ids(nx_graph, node)
        return PT2OpLayerAttributes(meta.func, meta.args, meta.kwargs, constant_port_ids)
    if isinstance(meta, ConstMeta):
        return ConstantLayerAttributes(meta.name_in_model, list(meta.shape))
    return None


def convert_to_nncf_graph(nx_graph: nx.MultiDiGraph) -> PTNNCFGraph:
    """
    Converts a graph to an PTNNCFGraph.

    :param nx_graph: The graph to convert.
    :return: The converted NNCFGraph.
    """
    nncf_graph = PTNNCFGraph()

    map_nx_node_to_nncf_node: dict[int, NNCFNode] = {}
    for node, data in nx_graph.nodes(data=True):
        meta = data["meta"]
        if not isinstance(meta, (ConstMeta, FunctionMeta, InOutMeta)):
            msg = f"Unknown metadata type: {type(meta)}"
            raise nncf.InternalError(msg)
        node_name = get_name_of_node(meta)
        node_type = get_node_type(data["type"], meta)
        meta_type = get_meta_type(node_type, meta)
        layer_attributes = get_layer_attributes(nx_graph, node, meta)
        nncf_node = nncf_graph.add_nncf_node(
            layer_attributes=layer_attributes,
            layer_name=node_name,
            node_metatype=meta_type,
            node_name=node_name,
            node_type=node_type,
        )
        map_nx_node_to_nncf_node[node] = nncf_node

    map_edges: dict[tuple[int, int], list[EdgeMeta]] = defaultdict(list)

    for s_node, t_node, data in nx_graph.edges(data=True):
        meta = data["meta"]
        if isinstance(meta, EdgeMeta):
            map_edges[(s_node, t_node)].append(meta)

    for (s_node, t_node), list_meta in map_edges.items():
        source_node = map_nx_node_to_nncf_node[s_node]
        target_node = map_nx_node_to_nncf_node[t_node]
        nncf_graph.add_edge_between_nncf_nodes(
            source_node.node_id,
            target_node.node_id,
            tensor_shape=list_meta[0].shape,
            input_port_id=list_meta[0].input_port,
            output_port_id=list_meta[0].output_port,
            dtype=get_dtype(list_meta[0].dtype),
            parallel_input_port_ids=[meta.input_port for meta in list_meta[1:]] if len(list_meta) > 1 else None,
        )
    return nncf_graph


def build_nncf_graph(model: nn.Module, example_input: Any) -> PTNNCFGraph:
    """
    Builds an NNCF graph from the given PyTorch model.

    :param model: The PyTorch model to build the graph from.
    :param example_input: An example input that will be used for model tracing.
    :return: The NNCF graph representation of the model.
    """
    if isinstance(example_input, dict):
        graph = build_graph(model, **example_input)
    elif isinstance(example_input, tuple):
        graph = build_graph(model, *example_input)
    else:
        graph = build_graph(model, example_input)
    return convert_to_nncf_graph(graph)


class GraphModelWrapper:
    """
    A class that wraps a PyTorch model with examples inputs and provides an interface
    to build a computational graph of the model.
    """

    def __init__(self, model: nn.Module, example_input: Any) -> None:
        """
        :param model: The PyTorch model to be wrapped.
        :param example_input: A tuple of example input for the model.
        """
        self.model = model
        self.example_input = example_input
        self.graph: PTNNCFGraph | None = None

    def build_graph(self) -> PTNNCFGraph:
        """
        Constructs a computational graph of the given model.

        This function builds a directed graph `PTNNCFGraph` representing the operations
        and data flow within the model by leveraging hooks by using GraphBuilderMode.

        :return: A PTNNCFGraph where nodes represent operations of model.
        """
        return build_nncf_graph(self.model, self.example_input)

    def get_graph(self) -> PTNNCFGraph:
        """
        Returns the computational graph of the model.

        :return: The PTNNCFGraph representing the model.
        """
        if self.graph is None:
            self.graph = self.build_graph()
        return self.graph

    def reset_graph(self) -> None:
        """
        Resets the computational graph of the model.
        """
        self.graph = None
