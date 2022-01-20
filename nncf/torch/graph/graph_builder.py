"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch

from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.graph import LayerName
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.utils import get_concat_axis
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_graph(self, model: torch.nn.Module, context_to_use: Optional['TracingContext'] = None,
                    as_eval: bool = False,
                    input_infos: List[ModelInputInfo] = None) -> PTNNCFGraph:
        tracer = GraphTracer(self.custom_forward_fn)
        dynamic_graph = tracer.trace_graph(model, context_to_use, as_eval)
        return GraphConverter.convert(dynamic_graph, input_infos)


class GraphConverter:
    @staticmethod
    def convert(dynamic_graph: DynamicGraph, input_infos: List[ModelInputInfo] = None) -> PTNNCFGraph:
        # pylint:disable=too-many-branches
        layer_name_vs_node_counts = {}  # type: Dict[LayerName, int]

        for dynamic_graph_node in dynamic_graph.get_all_nodes():
            layer_name = str(dynamic_graph_node.op_exec_context.op_address.scope_in_model)
            if layer_name not in layer_name_vs_node_counts:
                layer_name_vs_node_counts[layer_name] = 1
            else:
                layer_name_vs_node_counts[layer_name] += 1

        nncf_graph = PTNNCFGraph()
        for dynamic_graph_node in dynamic_graph.get_all_nodes():
            op_address = dynamic_graph_node.op_exec_context.op_address
            layer_name = str(dynamic_graph_node.op_exec_context.op_address.scope_in_model)

            metatype = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_address.operator_name)
            if (metatype is not UnknownMetatype and
                    not isinstance(dynamic_graph_node.layer_attributes, GenericWeightedLayerAttributes)):
                subtype = metatype.determine_subtype(dynamic_graph_node.layer_attributes)
            else:
                subtype = None
            if subtype is not None:
                metatype = subtype

            is_integer_input = False
            if metatype in INPUT_NOOP_METATYPES and input_infos is not None:
                input_id = op_address.call_order
                if input_infos[input_id].is_integer_input():
                    is_integer_input = True

            is_shared = layer_name in layer_name_vs_node_counts and layer_name_vs_node_counts[layer_name] > 1

            nncf_graph.add_nncf_node(node_name=str(op_address),
                                     node_type=op_address.operator_name,
                                     node_metatype=metatype,
                                     layer_attributes=dynamic_graph_node.layer_attributes,
                                     node_id_override=dynamic_graph_node.node_id,
                                     layer_name=str(op_address.scope_in_model),
                                     ignored_algorithms=dynamic_graph_node.ignored_algorithms,
                                     is_in_iteration_scope=dynamic_graph_node.is_in_iteration_scope,
                                     is_integer_input=is_integer_input,
                                     is_shared=is_shared)

        for dynamic_graph_edge in dynamic_graph.get_all_edges():
            nncf_graph.add_edge_between_nncf_nodes(
                from_node_id=dynamic_graph_edge.from_node_id,
                to_node_id=dynamic_graph_edge.to_node_id,
                tensor_shape=dynamic_graph_edge.activation_shape,
                input_port_id=dynamic_graph_edge.input_port_id,
                output_port_id=dynamic_graph_edge.output_port_id,
                dtype=dynamic_graph_edge.dtype
            )

        for node in nncf_graph.get_all_nodes():
            if node.metatype is PTCatMetatype:
                input_edges = nncf_graph.get_input_edges(node)
                output_edges = nncf_graph.get_output_edges(node)
                # Case of intermediate node
                if input_edges and output_edges:
                    input_shapes = [edge.tensor_shape for edge in input_edges]
                    output_shapes = [edge.tensor_shape for edge in output_edges]
                    # Case node is stack
                    if len(input_shapes[0]) != len(output_shapes[0]):
                        continue
                    axis = get_concat_axis(input_shapes, output_shapes)
                    layer_attributes = MultipleInputLayerAttributes(axis)
                    node.layer_attributes = layer_attributes

            if node.metatype is PTReshapeMetatype:
                input_nodes = nncf_graph.get_input_edges(node)
                output_nodes = nncf_graph.get_output_edges(node)
                # In case ReshapeMetatype op is intermediate node
                if input_nodes and output_nodes:
                    layer_attributes = ReshapeLayerAttributes(input_nodes[0].tensor_shape,
                                                              output_nodes[0].tensor_shape)
                    node.layer_attributes = layer_attributes
        return nncf_graph
