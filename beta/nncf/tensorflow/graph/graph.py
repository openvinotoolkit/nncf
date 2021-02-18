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

from typing import List

import networkx as nx

from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nxmodel
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.layers.common import GENERAL_CONV_LAYERS
from nncf.common.graph.graph import NNCFNode, NNCFGraph
from nncf.common.graph.module_attributes import ConvolutionModuleAttributes


class TFNNCFNode(NNCFNode):

    def __str__(self):
        return ' '.join([self.node_id, self.data[NNCFGraph.KEY_NODE_ATTR], self.node_type])

    def __eq__(self, other):
        return isinstance(other, TFNNCFNode) \
               and super().__eq__(other)

    @property
    def node_type(self):
        return self.data.get(TFNNCFGraph.NODE_TYPE_ATTR)


def tf_get_layer_identifier(node: NNCFNode):
    node_key = node.data[NNCFGraph.KEY_NODE_ATTR]
    layer_name, _ = get_original_name_and_instance_index(node_key)
    return layer_name


class TFNNCFGraph(NNCFGraph):

    NODE_TYPE_ATTR = 'type'

    def __init__(self, model):
        super().__init__()
        self._nx_graph = convert_keras_model_to_nxmodel(model)
        self._node_id_to_key_dict = self._provide_id_to_key_mapping()
        self._mark_graph_nodes_with_module_attributes(model)
        self._input_nncf_nodes = self._set_input_nodes()

    def _provide_id_to_key_mapping(self):
        node_id_to_key_dict = {}
        for id, key in enumerate(nx.topological_sort(self._nx_graph)):
            nx_node = self._nx_graph.nodes[key]
            nx_node[NNCFGraph.KEY_NODE_ATTR] = key
            nx_node[NNCFGraph.ID_NODE_ATTR] = id
            node_id_to_key_dict[id] = key
        return node_id_to_key_dict

    def _set_input_nodes(self) -> List[TFNNCFNode]:
        inputs = []
        for nx_node_key, deg in self._nx_graph.in_degree():
            if deg == 0:
                inputs.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[nx_node_key]))
        return inputs

    def _mark_graph_nodes_with_module_attributes(self, model):
        for node in self.get_nodes_by_types(GENERAL_CONV_LAYERS):
            for layer in model.layers:
                if layer.name == tf_get_layer_identifier(node):
                    module = layer
                    break

            nx_node = node.data

            channel_axis = -1 if nx_node['data_format'] == 'channels_last' else 1
            nx_node[self.MODULE_ATTRIBUTES] = ConvolutionModuleAttributes(module.trainable,
                                                                          module.input_shape[channel_axis],
                                                                          module.output_shape[channel_axis],
                                                                          module.strides[0],
                                                                          module.groups)

    @staticmethod
    def node_type_fn(node: dict) -> str:
        return node[TFNNCFGraph.NODE_TYPE_ATTR]

    @staticmethod
    def _nx_node_to_nncf_node(nx_node: dict) -> TFNNCFNode:
        return TFNNCFNode(nx_node[NNCFGraph.ID_NODE_ATTR],
                          nx_node)
