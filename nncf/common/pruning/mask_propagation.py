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

from typing import Dict, List

from nncf.common.graph import NNCFGraph
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import PruningAnalysisDecision
from nncf.common.pruning.utils import PruningAnalysisReason
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.operations import BasePruningOp


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

    def get_meta_operation_by_type_name(self, type_name: str) -> BasePruningOp:
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

    def symbolic_mask_propagation(self, prunable_layers_types: List[str],
                                  can_prune_after_analisys: Dict[int, PruningAnalysisDecision]) \
            -> Dict[int, PruningAnalysisDecision]:
        """
        Check all nodes marked as prunable after model analysis and pruning algo compatibility check
        have correspondent closing node, which means each prunable by output channels dimension convolution
        has correspondent prunable by input channels dimension convolution. Otherwise the whole group
        contained such node cannot be pruned.

        :param prunable_layers_types: Types of operations with prunable filters.
        :param can_prune_after_analisys: Dict of nodes indexes only indexes of convolutional
            layers that have no conflicts for MaskPropagation and supported by
            nncf pruning algorithm have True value.
        """

        can_be_closing_convs = set([node.node_id for node in self._graph.get_all_nodes()
                                    if node.node_type in prunable_layers_types and not is_grouped_conv(node)])
        can_prune_by_dim = {k: None for k in can_be_closing_convs}
        for node in self._graph.topological_sort():
            if node.node_id in can_be_closing_convs and can_prune_after_analisys[node.node_id]:
                # Set output mask
                node.data['output_mask'] = SymbolicMask(node.layer_attributes.out_channels, [node.node_id])
            # Propagate masks
            cls = self.get_meta_operation_by_type_name(node.node_type)
            cls.mask_propagation(node, self._graph)
            if node.node_id in can_be_closing_convs:
                # Check input mask producers
                input_masks = get_input_masks(node, self._graph)
                if any(input_masks):
                    assert len(input_masks) == 1
                    input_mask = input_masks[0]
                    if input_mask.mask_producers is None:
                        continue

                    for producer in input_mask.mask_producers:
                        previously_dims_equal = True if can_prune_by_dim[producer] is None \
                            else can_prune_by_dim[producer]

                        is_dims_equal = node.layer_attributes.in_channels == input_mask.shape[0]
                        decision = previously_dims_equal and is_dims_equal
                        can_prune_by_dim[producer] = PruningAnalysisDecision(
                            decision, PruningAnalysisReason.DIMENSION_MISMATCH)

        # Clean nodes masks
        for node in self._graph.get_all_nodes():
            node.data['output_mask'] = None

        convs_without_closing_conv = {}
        for k, v in can_prune_by_dim.items():
            if v is None:
                convs_without_closing_conv[k] = \
                    PruningAnalysisDecision(False, PruningAnalysisReason.CLOSING_CONV_MISSING)

        can_prune_by_dim.update(convs_without_closing_conv)
        return can_prune_by_dim
