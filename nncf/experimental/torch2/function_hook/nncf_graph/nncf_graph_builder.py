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


from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import networkx as nx  # type: ignore
import torch
from torch import nn

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph
from nncf.experimental.torch2.function_hook.graph.graph_utils import ConstMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import EdgeMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import FunctionMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import InOutMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import NodeType


def get_node_type(type: NodeType, meta: Union[ConstMeta, FunctionMeta, InOutMeta]) -> str:
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
        return meta.fn_name
    raise nncf.InternalError("Unexpected metadata type")


def get_name_of_node(meta: Union[ConstMeta, FunctionMeta, InOutMeta]) -> str:
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
    raise nncf.InternalError("Unexpected metadata type")


def get_dtype(dtype: torch.dtype) -> Dtype:
    """
    Converts a torch dtype to a NNCF Dtype enum.

    :param dtype: The torch dtype to be converted.
    :return: The corresponding NNCF Dtype enum value.
    """
    if dtype in [torch.float, torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return Dtype.FLOAT
    return Dtype.INTEGER


def get_meta_type(node_type: str, meta: Union[ConstMeta, FunctionMeta, InOutMeta]) -> om.PTOperatorMetatype:
    """
    Converts the node type and metadata into a PTOperatorMetatype object.
    :param node_type: The type of the node.
    :param meta: The metadata associated with the node.
    :return: The PTOperatorMetatype object.
    """
    node_metatype = cast(om.PTOperatorMetatype, om.PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type))
    node_sub_meta_type: Optional[om.PTOperatorMetatype] = None
    if node_metatype.get_subtypes() and isinstance(meta, FunctionMeta):
        node_sub_meta_type = node_metatype.determine_subtype(function_args=meta.args, functions_kwargs=meta.kwargs)
    return node_sub_meta_type or node_metatype


def convert_to_nncf_graph(nx_graph: nx.MultiDiGraph) -> NNCFGraph:
    """
    Converts a graph to an NNCFGraph.

    :param nx_graph: The graph to convert.
    :return: The converted NNCFGraph.
    """
    nncf_graph = NNCFGraph()

    map_nx_node_to_nncf_node: Dict[int, NNCFNode] = {}
    for node, data in nx_graph.nodes(data=True):
        meta: Union[ConstMeta, FunctionMeta, InOutMeta] = data["meta"]
        node_name = get_name_of_node(meta)
        node_type = get_node_type(data["type"], meta)
        meta_type = get_meta_type(node_type, meta)

        nncf_node = nncf_graph.add_nncf_node(
            node_name=node_name,
            node_type=node_type,
            node_metatype=meta_type,  # type: ignore[arg-type]
        )
        map_nx_node_to_nncf_node[node] = nncf_node

    map_edges: Dict[Tuple[int, int], List[EdgeMeta]] = defaultdict(list)

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


def build_nncf_graph(model: nn.Module, *args: Any, **kwargs: Any) -> NNCFGraph:
    """
    Builds an NNCF graph from the given PyTorch model.

    :param model: The PyTorch model to build the graph from.
    :param *args: Positional arguments to model inference.
    :param **kwargs: Keyword arguments to model inference.
    :return: The NNCF graph representation of the model.
    """
    graph = build_graph(model, *args, **kwargs)
    return convert_to_nncf_graph(graph)
