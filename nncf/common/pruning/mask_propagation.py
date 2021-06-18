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
from typing import List
from typing import Union
from typing import TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.export_helpers import DefaultMetaOp
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry

TensorType = TypeVar('TensorType')


class MaskPropagationAlgorithm:
    """
    Algorithm responsible for propagation masks across all nodes in the graph.
    Before call mask_propagation() you need set node.data['output_masks']
    for nodes that have masks already defined.
    """

    def __init__(self, graph: NNCFGraph, pruning_operator_metatypes: PruningOperationsMetatypeRegistry):
        """
        Initializes MaskPropagationAlgorithm.

        :param graph: Graph to work with.
        :param pruning_operator_metatypes: Registry with operation metatypes pruning algorithm is aware of, i.e.
               metatypes describing operations with common pruning mask application and propagation properties.
        """
        self._graph = graph
        self._pruning_operator_metatypes = pruning_operator_metatypes

    def get_meta_operation_by_type_name(self, type_name: str) -> DefaultMetaOp:
        """
        Returns class of metaop that corresponds to `type_name` type.

        :param type_name: Name of type of layer
        :return: Class of metaop that corresponds to `type_name` type.
        """
        cls = self._pruning_operator_metatypes.get_operator_metatype_by_op_name(type_name)
        if cls is None:
            cls = self._pruning_operator_metatypes.registry_dict['stop_propagation_ops']
        return cls

    def mask_propagation(self):
        """
        Mask propagation in graph:
        to propagate masks run method mask_propagation (of metaop of current node) on all nodes in topological order.
        """
        for node in self._graph.topological_sort():
            cls = self.get_meta_operation_by_type_name(node.node_type)
            cls.mask_propagation(node, self._graph)


def get_input_masks(node: NNCFNode, graph: NNCFGraph) -> List[Union[TensorType, None]]:
    """
    Returns input masks for all inputs of nx_node.

    :return: Input masks.
    """
    input_masks = [input_node.data['output_mask'] for input_node in graph.get_previous_nodes(node)]
    return input_masks


def identity_mask_propagation(node: NNCFNode, graph: NNCFGraph):
    """
    Propagates input mask through nx_node.
    """
    input_masks = get_input_masks(node, graph)
    assert len(input_masks) == 1
    node.data['input_masks'] = input_masks
    node.data['output_mask'] = input_masks[0]
