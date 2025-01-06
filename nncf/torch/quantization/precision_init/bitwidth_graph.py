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
from typing import Dict, Set, Tuple

import networkx as nx

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.torch.layers import NNCFConv2d
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import QuantizationController
from nncf.torch.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.torch.quantization.structs import NonWeightQuantizerInfo


class BitwidthGraph:
    def __init__(
        self,
        algo_ctrl: QuantizationController,
        model: NNCFNetwork,
        groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
        add_flops=False,
    ):
        nncf_graph = model.nncf.get_graph()
        self._nx_graph = nncf_graph.get_graph_for_structure_analysis()
        if add_flops:
            flops_per_module = model.nncf.get_flops_per_module()

            flops_vs_node_group: Dict[int, Tuple[int, Set[NNCFNode]]] = defaultdict(set)
            for idx, module_node_name_and_flops in enumerate(flops_per_module.items()):
                module_node_name, flops = module_node_name_and_flops
                node_set = set(nncf_graph.get_op_nodes_in_scope(nncf_graph.get_scope_by_node_name(module_node_name)))
                flops_vs_node_group[idx] = (flops, node_set)

        grouped_mode = bool(groups_of_adjacent_quantizers)
        for node_key, node in nncf_graph.nodes.items():
            color = ""
            operator_name = node.node_type
            module = model.nncf.get_containing_module(node.node_name)
            if isinstance(module, NNCFConv2d):
                color = "lightblue"
                if module.groups == module.in_channels and module.in_channels > 1:
                    operator_name = "DW_Conv2d"
                    color = "purple"
                kernel_size = "x".join(map(str, module.kernel_size))
                operator_name += f"_k{kernel_size}"
                padding_values = set(module.padding)
                padding_enabled = len(padding_values) >= 1 and padding_values.pop()
                if padding_enabled:
                    operator_name += "_PAD"
                if add_flops:
                    matches = [
                        f_nodes_tpl for idx, f_nodes_tpl in flops_vs_node_group.items() if node in f_nodes_tpl[1]
                    ]
                    assert len(matches) == 1
                    flops, affected_nodes = next(iter(matches))
                    operator_name += f"_FLOPS:{str(flops)}"
                    if len(affected_nodes) > 1:
                        node_ids = sorted([n.node_id for n in affected_nodes])
                        operator_name += "(shared among nodes {})".format(
                            ",".join([str(node_id) for node_id in node_ids])
                        )
            operator_name += "_#{}".format(node.node_id)
            target_node_to_draw = self._nx_graph.nodes[node_key]
            target_node_to_draw["label"] = operator_name
            target_node_to_draw["style"] = "filled"
            if color:
                target_node_to_draw["color"] = color

        non_weight_quantizers = algo_ctrl.non_weight_quantizers
        bitwidth_color_map = {2: "purple", 4: "red", 8: "green", 6: "orange"}
        for quantizer_id, quantizer_info in non_weight_quantizers.items():
            self._paint_activation_quantizer_node(
                nncf_graph, quantizer_id, quantizer_info, bitwidth_color_map, groups_of_adjacent_quantizers
            )
        for wq_id, wq_info in algo_ctrl.weight_quantizers.items():
            nodes = [nncf_graph.get_node_by_name(tp.target_node_name) for tp in wq_info.affected_insertions]
            if not nodes:
                raise AttributeError(
                    "Failed to get affected nodes for quantized module node: {}".format(wq_id.target_node_name)
                )
            preds = [nncf_graph.get_previous_nodes(node) for node in nodes]
            wq_nodes = []
            for pred_list in preds:
                for pred_node in pred_list:
                    if "UpdateWeight" in pred_node.node_name:
                        wq_nodes.append(pred_node)
            assert len(wq_nodes) == 1

            node = wq_nodes[0]
            node_id = node.node_id
            key = nncf_graph.get_node_key_by_id(node_id)
            nx_node_to_draw_upon = self._nx_graph.nodes[key]
            quantizer = wq_info.quantizer_module_ref
            bitwidths = quantizer.num_bits
            nx_node_to_draw_upon["label"] = "WFQ_[{}]_#{}".format(quantizer.get_quantizer_config(), str(node_id))
            if grouped_mode:
                group_id_str = "UNDEFINED"
                group_id = groups_of_adjacent_quantizers.get_group_id_for_quantizer(wq_id)
                if group_id is None:
                    nncf_logger.debug(f"No group for weight quantizer for: {wq_id}")
                else:
                    group_id_str = str(group_id)
                nx_node_to_draw_upon["label"] += "_G" + group_id_str
            nx_node_to_draw_upon["color"] = bitwidth_color_map[bitwidths]
            nx_node_to_draw_upon["style"] = "filled"

    def _paint_activation_quantizer_node(
        self,
        nncf_graph: NNCFGraph,
        quantizer_id: NonWeightQuantizerId,
        quantizer_info: NonWeightQuantizerInfo,
        bitwidth_color_map: Dict[int, str],
        groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
    ):
        affected_insertion_points_list = quantizer_info.affected_insertions

        for target_point in affected_insertion_points_list:
            nncf_node_name = target_point.target_node_name
            nncf_node = nncf_graph.get_node_by_name(nncf_node_name)
            node_id = nncf_node.node_id

            input_port_id = target_point.input_port_id

            if input_port_id is None:
                # Post-hooking used for activation quantization
                # Currently only a single post-hook can immediately follow an operation
                succs = list(nncf_graph.get_next_nodes(nncf_node))
                assert len(succs) == 1
                target_nncf_node_key = nncf_graph.get_node_key_by_id(succs[0].node_id)
            else:
                # Pre-hooking used for activation quantization
                previous_nodes = nncf_graph.get_previous_nodes(nncf_node)
                target_node = None
                for prev_node in previous_nodes:
                    prev_edge = nncf_graph.get_nx_edge(prev_node, nncf_node)
                    if prev_edge[NNCFGraph.INPUT_PORT_ID_EDGE_ATTR] == input_port_id:
                        target_node = prev_node
                        break

                assert target_node is not None, "Could not find a pre-hook quantizer node for a specific input port!"
                target_nncf_node_id = target_node.node_id
                target_nncf_node_key = nncf_graph.get_node_key_by_id(target_nncf_node_id)

            activation_fq_node = self._nx_graph.nodes[target_nncf_node_key]
            bitwidth = quantizer_info.quantizer_module_ref.num_bits
            activation_fq_node["color"] = bitwidth_color_map[bitwidth]
            activation_fq_node["style"] = "filled"
            node_id = activation_fq_node[NNCFNode.ID_NODE_ATTR]

            activation_fq_node["label"] = "AFQ_[{}]_#{}".format(
                quantizer_info.quantizer_module_ref.get_quantizer_config(), str(node_id)
            )
            grouped_mode = bool(groups_of_adjacent_quantizers)
            if grouped_mode:
                group_id_str = "UNDEFINED"
                group_id = groups_of_adjacent_quantizers.get_group_id_for_quantizer(quantizer_id)
                if node_id is None:
                    nncf_logger.debug(f"No group for activation quantizer: {target_nncf_node_key}")
                else:
                    group_id_str = str(group_id)
                activation_fq_node["label"] += "_G" + group_id_str

    def get(self) -> nx.DiGraph:
        return self._nx_graph
