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

from typing import Dict
from typing import List
from typing import Tuple
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PTNNCFGraph(NNCFGraph):
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

    def get_op_nodes_in_scope(self, scope: Scope) -> List[NNCFNode]:
        matching_graph_op_nodes = []
        for scope_str, nodes_in_module in self._layer_name_vs_shared_nodes.items():
            module_scope = Scope.from_str(scope_str)
            if module_scope in scope:
                matching_graph_op_nodes.extend(nodes_in_module)
        return matching_graph_op_nodes

    def get_scope_by_node_name(self, node_name: NNCFNodeName) -> Scope:
        matches = []
        for node_id, scope_str in self._node_ids_vs_layer_names.items():
            node = self.get_node_by_id(node_id)
            if node.node_name == node_name:
                matches.append(Scope.from_str(scope_str))
        assert len(matches) <= 1
        if not matches:
            raise RuntimeError("Node name {} not found in the node-vs-scope dict!".format(node_name))
        return matches[0]
