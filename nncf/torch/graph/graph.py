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

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import TypeVar

from nncf.common.graph.module_attributes import Dtype
from nncf.common.graph.graph import MODEL_INPUT_OP_NAME
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.module_attributes import BaseLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.graph_matching import find_subgraphs_match_expression


# pylint: disable=too-many-public-methods
from nncf.torch.graph.operator_metatypes import InputNoopMetatype
from nncf.torch.graph.operator_metatypes import NoopMetatype
from nncf.torch.graph.transformations.commands import PTTargetPoint

ModuleAttributes = TypeVar('ModuleAttributes', bound=BaseLayerAttributes)


class PTNNCFGraph(NNCFGraph):
    def __init__(self):
        super().__init__()
        self._node_vs_module_scope_dict = {}  # type: Dict[NNCFNode, Scope]
        self._module_scope_vs_nodes = defaultdict(list)  # type: Dict[Scope, List[NNCFNode]]

    @classmethod
    def from_dynamic_graph(cls, dynamic_graph: DynamicGraph, input_infos: List[ModelInputInfo] = None):
        nncf_graph = cls()
        for dynamic_graph_node in dynamic_graph.get_all_nodes():
            op_address = dynamic_graph_node.op_exec_context.op_address

            metatype = NoopMetatype  # TODO: fix
            is_integer_input = False
            if op_address.operator_name == MODEL_INPUT_OP_NAME and input_infos is not None:
                input_id = op_address.call_order
                if input_infos[input_id].is_integer_input():
                    is_integer_input = True
                metatype = InputNoopMetatype

            nncf_graph.add_nncf_node(node_name=str(op_address),
                                     node_type=op_address.operator_name,
                                     node_metatype=metatype,
                                     module_attributes=dynamic_graph_node.module_attributes,
                                     node_id_override=dynamic_graph_node.node_id,
                                     containing_module_scope=op_address.scope_in_model,
                                     ignored_algorithms=dynamic_graph_node.ignored_algorithms,
                                     is_in_iteration_scope=dynamic_graph_node.is_in_iteration_scope,
                                     is_integer_input=is_integer_input)

        for dynamic_graph_edge in dynamic_graph.get_all_edges():
            nncf_graph.add_edge_between_nncf_nodes(
                from_node_id=dynamic_graph_edge.from_node_id,
                to_node_id=dynamic_graph_edge.to_node_id,
                tensor_shape=dynamic_graph_edge.activation_shape,
                input_port_id=dynamic_graph_edge.input_port_id,
                dtype=Dtype.FLOAT
            )
        return nncf_graph

    def get_output_shapes_for_node(self, node_name: NNCFNodeName) -> List[Tuple]:
        node = self.get_node_by_name(node_name)
        node_key = self.get_node_key_by_id(node.node_id)
        succs = list(self._nx_graph.successors(node_key))
        edge_list = [self._nx_graph.edges[node_key, to_node_key] for to_node_key in succs]
        return [edge[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR] for edge in edge_list]

    def get_input_shapes_for_node(self, node_name: NNCFNodeName) -> Dict[int, Tuple]:
        node = self.get_node_by_name(node_name)
        node_key = self.get_node_key_by_id(node.node_id)
        in_edges = list(self._nx_graph.in_edges(node_key))
        retval = {}
        for in_edge in in_edges:
            edge_attr_dict = self._nx_graph.edges[in_edge]
            port_id = edge_attr_dict[NNCFGraph.IN_PORT_NAME_EDGE_ATTR]
            assert port_id not in retval
            retval[port_id] = edge_attr_dict[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
        return retval

    def get_input_shape_for_insertion_point(self, insertion_point: PTTargetPoint) -> Tuple[int]:
        target_node_name = insertion_point.target_node_name
        if insertion_point.input_port_id is not None:
            quantizer_input_shape = self.get_input_shapes_for_node(
                target_node_name)[insertion_point.input_port_id]
        else:
            # Tailored for post-hook quantization and first output quantization only
            quantizer_input_shape = self.get_output_shapes_for_node(
                target_node_name)[0]
        return quantizer_input_shape

    def get_op_nodes_in_scope(self, scope: 'Scope') -> List[NNCFNode]:
        matching_graph_op_nodes = []
        for module_scope, nodes_in_module in self._module_scope_vs_nodes.items():
            if module_scope in scope:
                matching_graph_op_nodes.extend(nodes_in_module)
        return matching_graph_op_nodes

    def add_nncf_node(self, node_name: str,
                      node_type: str,
                      node_metatype: Type[OperatorMetatype],
                      module_attributes: BaseLayerAttributes = None,
                      node_id_override: int = None,
                      containing_module_scope: Scope = None,
                      ignored_algorithms: List[str] = None,
                      is_in_iteration_scope: bool = False,
                      is_integer_input: bool = False) -> NNCFNode:
        node = super().add_nncf_node(node_name, node_type,
                                     node_metatype,
                                     module_attributes, node_id_override,
                                     ignored_algorithms, is_in_iteration_scope, is_integer_input)
        self._module_scope_vs_nodes[containing_module_scope].append(node)
        self._node_vs_module_scope_dict[node] = containing_module_scope
        return node

    def get_shared_nodes(self) -> List[List[NNCFNode]]:
        shared_nodes = list(x for x in self._module_scope_vs_nodes.values() if len(x) > 1)
        for idx, shared_node_group in enumerate(shared_nodes):
            # TODO: is this correct? Should shared nodes be only listed for corresponding weighted operations,
            # or for the entire node subgraph in the same scope as the weighted operation?
            shared_nodes[idx] = list(filter(lambda x: x.module_attributes is not None, shared_node_group))

        return shared_nodes

    def is_shared_node(self, node: NNCFNode) -> bool:
        scope = self._node_vs_module_scope_dict[node]
        nodes_sharing_scope = self._module_scope_vs_nodes[scope]
        return len(nodes_sharing_scope) > 1

    def get_scope_by_node_name(self, node_name: NNCFNodeName) -> Scope:
        matches = []
        for node, scope in self._node_vs_module_scope_dict.items():
            if node.node_name == node_name:
                matches.append(scope)
        assert len(matches) <= 1
        if not matches:
            raise RuntimeError("Node name {} not found in the node-vs-scope dict!".format(node_name))
        return matches[0]
