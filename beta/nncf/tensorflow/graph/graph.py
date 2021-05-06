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
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Type

from beta.nncf.tensorflow.graph.metatypes.common import OUTPUT_LAYER_METATYPES
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.module_attributes import BaseLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype


class TFNNCFGraph(NNCFGraph):
    def __init__(self):
        super().__init__()
        self._node_ids_vs_layer_names = {}  # type: Dict[NNCFNode, str]
        self._layer_name_vs_shared_nodes = defaultdict(list)  # type: Dict[str, List[NNCFNode]]

    def get_shared_nodes(self) -> List[List[NNCFNode]]:
        return [shared_nodes for shared_nodes in self._layer_name_vs_shared_nodes.values() if len(shared_nodes) > 1]

    def is_shared_node(self, node: NNCFNode) -> bool:
        containing_layer_name = self._node_ids_vs_layer_names[node.node_id]
        return len(self._layer_name_vs_shared_nodes[containing_layer_name]) > 1

    def add_nncf_node(self, node_name: str,
                      node_type: str,
                      node_metatype: Type[OperatorMetatype],
                      module_attributes: BaseLayerAttributes = None,
                      node_id_override: int = None,
                      containing_layer_name: str = None,
                      ignored_algorithms: List[str] = None,
                      is_in_iteration_scope: bool = False,
                      is_integer_input: bool = False) -> NNCFNode:
        node = super().add_nncf_node(node_name, node_type, node_metatype,
                                     module_attributes, node_id_override,
                                     ignored_algorithms=ignored_algorithms,
                                     is_in_iteration_scope=is_in_iteration_scope,
                                     is_integer_input=is_integer_input)
        if node_metatype not in OUTPUT_LAYER_METATYPES:
            self._layer_name_vs_shared_nodes[containing_layer_name].append(node)
        self._node_ids_vs_layer_names[node.node_id] = containing_layer_name
        return node
