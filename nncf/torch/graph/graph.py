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

from itertools import chain

import nncf
from nncf.common.check_features import is_torch_tracing_by_patching
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.function_hook.graph.graph_utils import TensorMeta
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PTNNCFGraph(NNCFGraph):
    def get_output_shapes_for_node(self, node_name: NNCFNodeName) -> list[tuple]:
        node = self.get_node_by_name(node_name)
        node_key = self.get_node_key_by_id(node.node_id)
        succs = list(self._nx_graph.successors(node_key))
        edge_list = [self._nx_graph.edges[node_key, to_node_key] for to_node_key in succs]
        return [edge[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR] for edge in edge_list]

    def get_input_shapes_for_node(self, node_name: NNCFNodeName) -> dict[int, tuple]:
        node = self.get_node_by_name(node_name)
        node_key = self.get_node_key_by_id(node.node_id)
        in_edges = list(self._nx_graph.in_edges(node_key))
        retval = {}
        for in_edge in in_edges:
            edge_attr_dict = self._nx_graph.edges[in_edge]
            port_id = edge_attr_dict[NNCFGraph.INPUT_PORT_ID_EDGE_ATTR]
            assert port_id not in retval
            for p in [
                port_id,
            ] + edge_attr_dict[NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR]:
                retval[p] = edge_attr_dict[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
        return retval

    def get_input_shape_for_insertion_point(self, insertion_point: PTTargetPoint) -> tuple[int]:
        target_node_name = insertion_point.target_node_name
        if insertion_point.input_port_id is not None:
            quantizer_input_shape = self.get_input_shapes_for_node(target_node_name)[insertion_point.input_port_id]
        else:
            # Tailored for post-hook quantization and first output quantization only
            quantizer_input_shape = self.get_output_shapes_for_node(target_node_name)[0]
        return quantizer_input_shape

    def get_op_nodes_in_scope(self, scope: Scope) -> list[NNCFNode]:
        """
        Returns all NNCFNodes inside the given scope.

        :param scope: Given scope.
        :return: All NNCFNodes inside the given scope.
        """
        matching_graph_op_nodes = []
        for scope_str, nodes_in_module in self._layer_name_vs_shared_nodes.items():
            module_scope = Scope.from_str(scope_str)
            if module_scope in scope:
                matching_graph_op_nodes.extend(nodes_in_module)
        return matching_graph_op_nodes

    def get_op_nodes_with_scope(self, scope: Scope) -> list[NNCFNode]:
        """
        Returns all NNCFNodes which share the given scope.

        :param scope: Given scope.
        :return: All NNCFNodes which share the given scope.
        """
        return self._layer_name_vs_shared_nodes[str(scope)]

    def get_scope_by_node_name(self, node_name: NNCFNodeName) -> Scope:
        """
        Returns a scope which corresponds to the given NNCF node name.

        :param node_name: Given node name.
        :return: A scope which corresponds to the given NNCF node name.
        """
        matches = []
        for node_id, scope_str in self._node_ids_vs_layer_names.items():
            node = self.get_node_by_id(node_id)
            if node.node_name == node_name:
                matches.append(Scope.from_str(scope_str))
        assert len(matches) <= 1
        if not matches:
            msg = f"Node name {node_name} not found in the node-vs-scope dict!"
            raise nncf.InternalError(msg)
        return matches[0]

    def get_nodes_with_missed_input_edges(self) -> list[NNCFNode]:
        """
        Returns a list of NNCFNodes that have at least one expected input edge missed.
        Requires MultipleInputLayerAttributes for nodes with several inputs and
        right `num_expected_input_edges` parameter setted for nncf nodes metatypes.

        :return: List of NNCFNodes that are identified as disconnected.
        """
        input_nodes = set()
        if is_torch_tracing_by_patching():
            for node in self.get_all_nodes():
                num_expected_input_edges = None
                if hasattr(node.metatype, "num_expected_input_edges"):
                    num_expected_input_edges = node.metatype.num_expected_input_edges
                if node.layer_attributes is not None and isinstance(
                    node.layer_attributes, MultipleInputLayerAttributes
                ):
                    num_expected_input_edges = node.layer_attributes.num_inputs
                if num_expected_input_edges:
                    input_edges = self.get_input_edges(node)
                    if len(input_edges) < num_expected_input_edges:
                        # If node has missed input edges we assume this node is an input node
                        # that was disconnected from an activation input.
                        input_nodes.add(node)
        else:
            # Check expected number of input edges by counting TensorMeta in op_args and op_kwargs.
            for node in self.get_all_nodes():
                input_edges = len(self.get_input_edges(node))
                if not isinstance(node.layer_attributes, PT2OpLayerAttributes):
                    continue
                num_expected_input_edges = 0
                for val in chain(node.layer_attributes.op_args, node.layer_attributes.op_kwargs.values()):
                    if isinstance(val, TensorMeta):
                        num_expected_input_edges += 1
                    if isinstance(val, (list, tuple)):
                        num_expected_input_edges += sum(isinstance(v, TensorMeta) for v in val)
                if input_edges < num_expected_input_edges:
                    input_nodes.add(node)

        return list(input_nodes)
