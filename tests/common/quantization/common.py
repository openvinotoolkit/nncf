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

from typing import List

from tests.common.quantization.metatypes import TestMetatype
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from collections import Counter


class NodeWithType:
    def __init__(self, name: str, op_type: TestMetatype):
        self.node_name = name
        self.node_op_type = op_type


class NNCFGraphToTest:
    def __init__(self, nodes: List[NodeWithType], node_edges):
        self.nncf_graph = NNCFGraph()
        for node in nodes:
            self.nncf_graph.add_nncf_node(node_name=node.node_name,
                                          node_type=node.node_op_type.name,
                                          node_metatype=node.node_op_type,
                                          layer_attributes=None)
        input_port_counter = Counter()
        output_port_counter = Counter()
        for from_node, to_nodes in node_edges.items():
            output_node_id = self.nncf_graph.get_node_by_name(from_node).node_id
            for to_node in to_nodes:
                input_node_id = self.nncf_graph.get_node_by_name(to_node).node_id
                self.nncf_graph.add_edge_between_nncf_nodes(output_node_id, input_node_id, [1],
                                                            input_port_counter[input_node_id],
                                                            output_port_counter[output_node_id], Dtype.FLOAT)
                input_port_counter[input_node_id] += 1
                output_port_counter[output_node_id] += 1
