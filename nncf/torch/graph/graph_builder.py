"""
 Copyright (c) 2021 Intel Corporation
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

from typing import Any, Callable, Optional

import torch

from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTNNCFNode
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_graph(self, model: torch.nn.Module, context_to_use: Optional['TracingContext'] = None,
                    as_eval: bool = False) -> PTNNCFGraph:
        tracer = GraphTracer(self.custom_forward_fn)
        traced_graph = tracer.trace_graph(model, context_to_use, as_eval)

        nncf_graph = PTNNCFGraph()
        for dynamic_graph_node in traced_graph.get_all_nodes():
            op_name = dynamic_graph_node.op_exec_context.operator_name
            op_metatype = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
            subtype = op_metatype.determine_subtype(dynamic_graph_node.module_attributes)
            if subtype is not None:
                op_metatype = subtype

            nncf_node = PTNNCFNode(
                node_id=dynamic_graph_node.node_id,
                ia_op_exec_context=dynamic_graph_node.op_exec_context.input_agnostic,
                data={
                    PTNNCFGraph.MODULE_ATTRIBUTES: dynamic_graph_node.module_attributes,
                    PTNNCFGraph.METATYPE_ATTR: op_metatype
                }
            )
            nncf_graph.add_nncf_node(nncf_node)

        for dynamic_graph_edge in traced_graph.get_all_edges():
            nncf_graph.add_edge_between_nncf_nodes(
                from_node_id=dynamic_graph_edge.from_node_id,
                to_node_id=dynamic_graph_edge.to_node_id,
                tensor_shape=dynamic_graph_edge.activation_shape,
                input_port_id=dynamic_graph_edge.input_port_id
            )
        return nncf_graph
