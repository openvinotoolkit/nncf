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
from typing import Optional

import networkx as nx

import nncf
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.pruning.utils import get_input_masks
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleDepthwiseConv2dSubtype


class SubnetGraph:
    """
    Graph that represents active subnet in convenient way for visualization.
    """

    def __init__(self, compression_graph: PTNNCFGraph, multi_elasticity_handler: MultiElasticityHandler):
        # TODO: visualize other elastic dimension: depth, kernel (ticket 76870)
        self._width_graph = compression_graph.get_graph_for_structure_analysis(extended=True)
        for node_key, compression_node in compression_graph.nodes.items():
            operator_name = self._get_operator_name(compression_node, multi_elasticity_handler.width_handler)

            metatype = compression_node.metatype
            color = None
            if metatype == PTModuleConv2dMetatype:
                color = "lightblue"
            if metatype == PTModuleDepthwiseConv2dSubtype:
                operator_name = f"DW_{operator_name}"
                color = "purple"

            target_node_to_draw = self._width_graph.nodes[node_key]
            target_node_to_draw["label"] = operator_name
            target_node_to_draw["style"] = "filled"
            if color is not None:
                target_node_to_draw["color"] = color

    def get(self) -> nx.DiGraph:
        return self._width_graph

    @staticmethod
    def _get_operator_name(compressed_node: NNCFNode, width_handler: ElasticWidthHandler):
        operator_name = compressed_node.node_type
        node_id = compressed_node.node_id
        node = SubnetGraph._get_original_node_from_compressed_node_name(compressed_node.node_name, width_handler)
        if node is not None:
            operator_name = node.node_type
            node_id = node.node_id
            input_masks = get_input_masks(node, width_handler.propagation_graph)
            input_widths = None
            if input_masks:
                input_widths = [ElasticWidthHandler.mask_to_width(input_mask) for input_mask in input_masks]
            output_width = ElasticWidthHandler.mask_to_width(node.attributes["output_mask"])

            if input_widths:
                IW = None
                if len(input_widths) == 1 and input_widths[0]:
                    IW = input_widths[0]
                if len(input_widths) > 1 and all(input_widths):
                    IW = input_widths
                if IW is not None:
                    operator_name += f"_IW{IW}"

            if output_width is not None:
                operator_name += f"_OW{output_width}"

            group_id = width_handler.get_group_id_by_node_name(node.node_name)
            if group_id is not None:
                operator_name += f"_G{group_id}"

        operator_name += f"_#{node_id}"
        return operator_name

    @staticmethod
    def _get_original_node_from_compressed_node_name(
        node_name: NNCFNodeName, width_handler: ElasticWidthHandler
    ) -> Optional[NNCFNode]:
        try:
            propagation_graph: PTNNCFGraph = width_handler.propagation_graph
            result = propagation_graph.get_node_by_name(node_name)
        except nncf.InternalError:
            result = None
        return result
