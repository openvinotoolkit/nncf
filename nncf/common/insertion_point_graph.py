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
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Set

import networkx as nx  # type: ignore

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.patterns import GraphPattern


class InsertionPointGraphNodeType(Enum):
    PRE_HOOK = 0
    POST_HOOK = 1
    OPERATOR = 2


class PreHookInsertionPoint:
    def __init__(self, target_node_name: str, input_port_id: int):
        self.target_node_name = target_node_name
        self.input_port_id = input_port_id

    def __str__(self) -> str:
        return str(self.input_port_id) + " " + self.target_node_name


class PostHookInsertionPoint:
    def __init__(self, target_node_name: str):
        self.target_node_name = target_node_name

    def __str__(self) -> str:
        return self.target_node_name


class InsertionPointGraph(nx.DiGraph):  # type: ignore
    """
    This graph is built from the NNCFGraph representation of the model control flow graph and adds ephemeral
    "insertion point nodes" into the NNCF model graph representation corresponding to operator pre- and
    post-hooks. Module pre-op and post-op insertion points are currently not reflected here, but they are
    probably not required for quantizing activations, for which the quantizer propagation makes sense.
    This "insertion point graph" representation is useful for quantizer propagation and for referencing
    the compression algorithm hooks to the model operations to which they are applied to.
    """

    NODE_TYPE_NODE_ATTR = "node_type"
    INSERTION_POINT_NODE_ATTR = "insertion_point"
    IS_IN_NNCF_MODULE_NODE_ATTR = "is_in_nncf_module"
    REGULAR_NODE_REF_NODE_ATTR = "regular_node_data"
    ASSOCIATED_IP_NODE_KEYS_NODE_ATTR = "associated_ip_node_keys"
    IS_MERGED_NODE_ATTR = "is_merged"
    MERGED_NNCF_NODE_LIST_NODE_ATTR = "merged_node_list"
    IS_INTEGER_PATH_EDGE_ATTR = "is_integer"

    PRE_HOOK_ID_PREFIX = "PRE HOOK "  # NB: Do not use colon (':') in node keys! Causes trouble for .dot file export.
    POST_HOOK_ID_PREFIX = "POST HOOK "

    def __init__(
        self,
        nncf_graph: NNCFGraph,
        allowed_pre_hook_insertion_points: List[PreHookInsertionPoint] = None,
        allowed_post_hook_insertion_points: List[PostHookInsertionPoint] = None,
    ):
        """
        Initializes the insertion point graph.

        :param nncf_graph: The base NNCFGraph representing the model structure.
        :param allowed_pre_hook_insertion_points: A list of pre-hook insertion points for this graph to allow.
          If left unspecified, every node in `nncf_graph` will be allowed to have a separate pre-hook for each of its
          tensor inputs.
        :param allowed_post_hook_insertion_points: A list of post-hook insertion points for this graph to allow.
        If left unspecified, every node in `nncf_graph` will be allowed to have a single post-hook for its output
         (post-hooking separate tensors in an operation's output is not currently supported)
        """

        super().__init__()
        self._base_nx_graph = deepcopy(nncf_graph.get_nx_graph_copy())

        if allowed_pre_hook_insertion_points is None:
            allowed_pre_hook_insertion_points = self._get_default_pre_hook_ip_list(nncf_graph)

        if allowed_post_hook_insertion_points is None:
            # Post-hook all nodes if an exact list is not specified
            allowed_post_hook_insertion_points = self._get_default_post_hook_ip_list(nncf_graph)

        target_node_name_vs_pre_hook_ips: Dict[NNCFNodeName, Set[PreHookInsertionPoint]] = defaultdict(set)
        for pre_hook_ip in allowed_pre_hook_insertion_points:
            target_node_name_vs_pre_hook_ips[pre_hook_ip.target_node_name].add(pre_hook_ip)

        target_node_name_vs_post_hook_ips: Dict[NNCFNodeName, Set[PostHookInsertionPoint]] = defaultdict(set)
        for post_hook_ip in allowed_post_hook_insertion_points:
            target_node_name_vs_post_hook_ips[post_hook_ip.target_node_name].add(post_hook_ip)

        for node_key in nx.lexicographical_topological_sort(self._base_nx_graph):
            nncf_node = nncf_graph.get_node_by_key(node_key)
            attrs = {
                InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR: nncf_node,
                InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.OPERATOR,
                InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR: set(),
                InsertionPointGraph.IS_MERGED_NODE_ATTR: False,
            }
            self.add_node(node_key, **attrs)

        INPUT_PORT_ID = "input_port_id"
        for edge in self._base_nx_graph.edges:
            input_port_id = self._base_nx_graph.edges[edge][NNCFGraph.INPUT_PORT_ID_EDGE_ATTR]
            dtype = self._base_nx_graph.edges[edge][NNCFGraph.DTYPE_EDGE_ATTR]
            parallel_input_port_ids = self._base_nx_graph.edges[edge][NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR]
            from_node, to_node = edge

            attrs = {
                INPUT_PORT_ID: input_port_id,
                self.IS_INTEGER_PATH_EDGE_ATTR: dtype is Dtype.INTEGER,
                NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR: parallel_input_port_ids,
            }
            self.add_edge(from_node, to_node, **attrs)

        node_keys_working_set = [deepcopy(node_key) for node_key in nx.lexicographical_topological_sort(self)]

        # TODO (vshampor): Add insertion points for module pre- and post-ops.
        # Should roughly look so: first, determine subsets of nodes belonging to each
        # separate NNCF module (via scope analysis), then for each subset find input/output
        # edges using a corresponding NNCFGraph function; add a pre-op insertion point node as the
        # sink for input edges and connect it to input edge destinations, then add a post-op
        # insertion point as the source of output edges and connect it to output edge origins.

        for operator_node_key in node_keys_working_set:
            operator_node = self.nodes[operator_node_key]
            original_node = operator_node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
            if original_node.node_name in target_node_name_vs_pre_hook_ips:
                pre_hook_ips = list(target_node_name_vs_pre_hook_ips[original_node.node_name])
                pre_hook_ips = sorted(pre_hook_ips, key=lambda x: x.input_port_id)
                in_edges = list(self.in_edges(operator_node_key))
                input_port_id_vs_edge = {}
                for edge in in_edges:
                    input_port_id = self.edges[edge][INPUT_PORT_ID]
                    input_port_id_vs_edge[input_port_id] = edge
                    for parallel_input_port_id in self.edges[edge][NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR]:
                        input_port_id_vs_edge[parallel_input_port_id] = edge

                encountered_input_edges = set()
                for pre_hook_point in pre_hook_ips:
                    edge = input_port_id_vs_edge[pre_hook_point.input_port_id]
                    original_edge_attrs = self.edges[edge]
                    from_node_key, to_node_key = edge
                    ip_node_key = self.get_pre_hook_node_key(str(operator_node_key), pre_hook_point.input_port_id)

                    pre_hook_ip_attrs = {
                        InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.PRE_HOOK,
                        InsertionPointGraph.INSERTION_POINT_NODE_ATTR: pre_hook_point,
                    }

                    self.add_node(ip_node_key, **pre_hook_ip_attrs)

                    encountered_input_edges.add(edge)
                    self.add_edge(from_node_key, ip_node_key, **original_edge_attrs)
                    self.add_edge(ip_node_key, operator_node_key, **original_edge_attrs)
                    operator_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)

                for edge in encountered_input_edges:
                    self.remove_edge(*edge)

            if original_node.node_name in target_node_name_vs_post_hook_ips:
                post_hook_ips = target_node_name_vs_post_hook_ips[original_node.node_name]
                assert len(post_hook_ips) == 1, "Multiple post-hooks for a single NNCFGraph node are not supported!"
                post_hook_ip = next(iter(post_hook_ips))
                post_hook_ip_attrs = {
                    InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.POST_HOOK,
                    InsertionPointGraph.INSERTION_POINT_NODE_ATTR: post_hook_ip,
                }
                ip_node_key = self.get_post_hook_node_key(str(operator_node_key))
                self.add_node(ip_node_key, **post_hook_ip_attrs)
                out_edges = list(self.out_edges(operator_node_key))
                has_integer_outputs = False
                for out_edge in out_edges:
                    # Need to preserve original edge attributes in order not to lose
                    # input port ID information
                    original_edge_attrs = self.edges[out_edge]
                    from_node_key, to_node_key = out_edge
                    self.remove_edge(from_node_key, to_node_key)
                    self.add_edge(ip_node_key, to_node_key, **original_edge_attrs)
                    if original_edge_attrs[self.IS_INTEGER_PATH_EDGE_ATTR]:
                        has_integer_outputs = True

                    # TODO (vshampor): introduce separate insertion points for operator outputs if
                    # the outputs are semantically different

                # TODO (vshampor): in multi-output case, some outputs may be integer and some float;
                #  need to switch to using output ports to cover this correctly. For safety, mark
                #  the edge from op to post-hook as integer if at least one output edge of the op was integer
                is_integer_attrs = {InsertionPointGraph.IS_INTEGER_PATH_EDGE_ATTR: has_integer_outputs}
                self.add_edge(operator_node_key, ip_node_key, **is_integer_attrs)
                operator_node = self.nodes[operator_node_key]
                operator_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)

        for edge in self.edges:
            # Mark all edges from post-hook to pre-hook as integer if at least one was integer.
            # Until output ports are ready, the post-hook for output will treat op as having a single
            # tensor output. In multi-output case when some of tensors are integer, need to make
            # sure that the propagation won't happen from a pre-hook of the op consuming the floating part
            # of the output into the post-hook of the operation that produces both int and float tensors.
            from_node_key, to_node_key = edge
            from_node = self.nodes[from_node_key]
            to_node = self.nodes[to_node_key]
            if (
                from_node[self.NODE_TYPE_NODE_ATTR] is InsertionPointGraphNodeType.POST_HOOK
                and to_node[self.NODE_TYPE_NODE_ATTR] is InsertionPointGraphNodeType.PRE_HOOK
            ):
                post_hook_has_integer_outputs = False
                for follower_node_key in self.successors(from_node_key):
                    if self.edges[from_node_key, follower_node_key][self.IS_INTEGER_PATH_EDGE_ATTR]:
                        post_hook_has_integer_outputs = True
                if post_hook_has_integer_outputs:
                    for follower_node_key in self.successors(from_node_key):
                        self.edges[from_node_key, follower_node_key][self.IS_INTEGER_PATH_EDGE_ATTR] = True

    @staticmethod
    def _get_default_pre_hook_ip_list(nncf_graph: NNCFGraph) -> List[PreHookInsertionPoint]:
        # Pre-hook all input ports of all nodes
        allowed_pre_hook_insertion_points = []
        for nncf_node in nncf_graph.get_all_nodes():
            pred_nodes = nncf_graph.get_previous_nodes(nncf_node)

            for pred_node in pred_nodes:
                input_edge = nncf_graph.get_edge(pred_node, nncf_node)
                allowed_pre_hook_insertion_points.append(
                    PreHookInsertionPoint(nncf_node.node_name, input_edge.input_port_id)
                )
        return allowed_pre_hook_insertion_points

    @staticmethod
    def _get_default_post_hook_ip_list(nncf_graph: NNCFGraph) -> List[PostHookInsertionPoint]:
        # Post-hook all nodes, post hook applies to the entire op output
        allowed_post_hook_insertion_points = []
        for nncf_node in nncf_graph.get_all_nodes():
            allowed_post_hook_insertion_points.append(PostHookInsertionPoint(nncf_node.node_name))
        return allowed_post_hook_insertion_points

    def remove_nodes_from(self, nodes: List[str]) -> None:
        """
        Removes nodes from the InsertionPointGraph and its _base_nx_graph.

        :param nodes: List of the nodes to remove.
        :return:
        """
        super().remove_nodes_from(nodes)
        constant_nodes_base_nx_graph = [node for node in nodes if node in self._base_nx_graph]
        self._base_nx_graph.remove_nodes_from(constant_nodes_base_nx_graph)

    def get_base_nx_graph(self) -> nx.DiGraph:
        """
        Returns the self._base_nx_graph.

        :return: The self._base_nx_graph.
        """
        return self._base_nx_graph

    def get_input_nodes(self) -> List[str]:
        """
        Returns all input nodes, meaning the nodes which belong to any of INPUT_NOOP_METATYPES metatype.

        :return: A list of input nodes.
        """
        output = []
        for node, data in self.nodes.items():
            if InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR not in data:
                continue
            if data[InsertionPointGraph.IS_MERGED_NODE_ATTR]:
                for nncf_node in data[InsertionPointGraph.MERGED_NNCF_NODE_LIST_NODE_ATTR]:
                    if self._base_nx_graph.nodes[nncf_node.node_key][NNCFNode.METATYPE_ATTR] in INPUT_NOOP_METATYPES:
                        output.append(node)
                        break
            elif data[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR].metatype in INPUT_NOOP_METATYPES:
                output.append(node)
        return output

    def get_merged_node_from_single_node_key(self, node_key: str) -> str:
        """
        Returns the node key of the composite node,
        if the corresponding node with 'node_key' was merged into it during the graph initialization.
        Otherwise, returns the original 'node_key'.

        :param node_key: The key of the node which is checking on the merged.
        :return: The node key of the composite node. Original 'node_key' if the node was not merged.
        """
        for node, data in self.nodes.items():
            if InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR not in data:
                continue
            if data[InsertionPointGraph.IS_MERGED_NODE_ATTR]:
                for nncf_node in data[InsertionPointGraph.MERGED_NNCF_NODE_LIST_NODE_ATTR]:
                    if node_key == nncf_node.node_key:
                        return node  # type: ignore
        return node_key

    def get_ip_graph_with_merged_hw_optimized_operations(
        self, full_fusing_pattern: GraphPattern
    ) -> "InsertionPointGraph":
        """
        Returns an InsertionPointGraph in which the nodes that match a HW-specific list of patterns are fused into a
        single node; the resulting InsertionPointGraph no longer has accessible the pre- and post-hooks that were
        located in  the middle of the fused pattern.
        If the InsertionPointGraph should be filtered from constant nodes before the node fusing,
        then 'known_non_constant_node_keys' should be pass. This is the list of the node known that are non constansts.

        :param full_fusing_pattern: The GraphPatttern object representing a composition of fusing pattern variants.
        :return: The InsertionPointGraph with nodes fused according to pattern matching.
        """

        merged_ip_graph = deepcopy(self)
        matches = find_subgraphs_matching_pattern(merged_ip_graph.get_base_nx_graph(), full_fusing_pattern)
        for match in matches:
            if len(match) == 1:
                continue

            input_node_key = match[0]
            output_node_key = match[-1]

            in_edges = list(self.in_edges(input_node_key))
            out_edges = list(self.out_edges(output_node_key))

            in_edge_copies_dict = {}
            for in_edge_key in in_edges:
                in_edge_copies_dict[in_edge_key] = deepcopy(self.edges[in_edge_key])
            out_edge_copies_dict = {}
            for out_edge_key in out_edges:
                out_edge_copies_dict[out_edge_key] = deepcopy(self.edges[out_edge_key])

            conserved_edges_list = out_edges + in_edges

            merged_node_attrs = deepcopy(self.nodes[input_node_key])
            merged_node_attrs[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR] = set()
            merged_node_attrs[InsertionPointGraph.IS_MERGED_NODE_ATTR] = True
            merged_node_key = ""
            merged_nncf_nodes = []
            for node_key in match:
                ip_node_keys = self.nodes[node_key][InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]
                for ip_node_key in ip_node_keys:
                    should_keep_ip_node = False
                    for edge_key in conserved_edges_list:
                        if ip_node_key in edge_key:
                            should_keep_ip_node = True
                            break
                    if should_keep_ip_node:
                        merged_node_attrs[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)
                    else:
                        merged_ip_graph.remove_node(ip_node_key)
                merged_nncf_nodes.append(self.nodes[node_key][InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR])
                merged_ip_graph.remove_node(node_key)
                merged_node_key += node_key + "\n"

            # The first node in the merged node list will be considered a "primary" node for purposes
            # of further ignored/target scope application.
            merged_node_attrs[InsertionPointGraph.MERGED_NNCF_NODE_LIST_NODE_ATTR] = merged_nncf_nodes
            merged_ip_graph.add_node(merged_node_key, **merged_node_attrs)
            for in_edge_key, in_edge_attrs in in_edge_copies_dict.items():
                merged_ip_graph.add_edge(in_edge_key[0], merged_node_key, **in_edge_attrs)
            for out_edge_key, out_edge_attrs in out_edge_copies_dict.items():
                merged_ip_graph.add_edge(merged_node_key, out_edge_key[1], **out_edge_attrs)

        return merged_ip_graph

    @staticmethod
    def get_pre_hook_node_key(node_key: str, input_port_id: int = 0) -> str:
        return InsertionPointGraph.PRE_HOOK_ID_PREFIX + str(input_port_id) + " " + node_key

    @staticmethod
    def get_post_hook_node_key(node_key: str) -> str:
        return InsertionPointGraph.POST_HOOK_ID_PREFIX + node_key
