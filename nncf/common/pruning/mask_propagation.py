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

from typing import Dict, List, Optional, Type

from nncf.common.graph import NNCFGraph
from nncf.common.pruning.operations import BasePruningOp
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMaskProcessor
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import PruningAnalysisDecision
from nncf.common.pruning.utils import PruningAnalysisReason
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import get_input_channels
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import get_output_channels
from nncf.common.pruning.utils import is_batched_linear
from nncf.common.pruning.utils import is_grouped_conv


class MaskPropagationAlgorithm:
    """
    Algorithm responsible for propagation masks across all nodes in the graph.
    Before call mask_propagation() you need set node.attributes['output_masks']
    for nodes that have masks already defined.
    """

    def __init__(
        self,
        graph: NNCFGraph,
        pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
        tensor_processor: Optional[Type[NNCFPruningBaseTensorProcessor]] = None,
    ):
        """
        Initializes MaskPropagationAlgorithm.

        :param graph: Graph to work with.
        :param pruning_operator_metatypes: Registry with operation metatypes pruning algorithm is aware of, i.e.
               metatypes describing operations with common pruning mask application and propagation properties.
        :param tensor_processor: Framework-specific tensor processor.
        """
        self._graph = graph
        self._pruning_operator_metatypes = pruning_operator_metatypes
        self._tensor_processor = tensor_processor

    def get_meta_operation_by_type_name(self, type_name: str) -> BasePruningOp:
        """
        Returns class of metaop that corresponds to `type_name` type.

        :param type_name: Name of type of layer
        :return: Class of metaop that corresponds to `type_name` type.
        """
        cls = self._pruning_operator_metatypes.get_operator_metatype_by_op_name(type_name)
        if cls is None:
            cls = self._pruning_operator_metatypes.registry_dict["stop_propagation_ops"]
        return cls

    def mask_propagation(self):
        """
        Mask propagation in graph:
        to propagate masks run method mask_propagation (of metaop of current node) on all nodes in topological order.
        """
        for node in self._graph.topological_sort():
            cls = self.get_meta_operation_by_type_name(node.node_type)
            cls.mask_propagation(node, self._graph, self._tensor_processor)

    def symbolic_mask_propagation(
        self, prunable_layers_types: List[str], can_prune_after_analysis: Dict[int, PruningAnalysisDecision]
    ) -> Dict[int, PruningAnalysisDecision]:
        """
        Check all nodes that were marked as prunable after the model analysis and compatibility check vs.
        pruning algo have a correct correspondent closing node on each path from self to outputs;
        the check entails verifying that every convolution prunable by the output channel dimension
        has a corresponding convolution that is prunable by its input channel dimension (output channel
        dimension equal to closing convolution input channel dimension) in every path from self to outputs.
        If the check fails, the entire groups containing such nodes will be marked as unprunable.
        If convolution symbolic mask mixes with other symbolic masks (by elementwise operation, for example)
        and mixing masks can't be mixed, all mask producers participated in this mixing will be marked as unprunable.
        If convolution output channel dimension reducing directly affect an output of the model -
        it will be marked as unprunable as well.


        :param prunable_layers_types: Types of operations with prunable filters.
        :param can_prune_after_analysis: Dict of node indices vs the decision made by previous steps;
            the decision is true only for the nodes that do not conflict with mask propagation and
            are supported by the NNCF pruning algorithm.
        :return: Dict of node indices vs the decision made by symbolic mask propagation algorithm.
        """

        can_be_closing_convs = self._get_can_closing_convs(prunable_layers_types)
        can_prune_by_dim = {k: None for k in can_be_closing_convs}
        for node in self._graph.topological_sort():
            if node.node_id in can_be_closing_convs and can_prune_after_analysis[node.node_id]:
                # Set output mask
                node.attributes["output_mask"] = SymbolicMask(get_output_channels(node), node.node_id)
            # Propagate masks
            cls = self.get_meta_operation_by_type_name(node.node_type)
            cls.mask_propagation(node, self._graph, SymbolicMaskProcessor)
            if node.node_id in can_be_closing_convs:
                # Check input mask producers out channel dimension
                input_masks = get_input_masks(node, self._graph)
                if any(input_masks):
                    assert len(input_masks) == 1
                    input_mask: SymbolicMask = input_masks[0]

                    for producer in input_mask.mask_producers:
                        previously_dims_equal = (
                            True if can_prune_by_dim[producer.id] is None else can_prune_by_dim[producer.id]
                        )

                        is_dims_equal = get_input_channels(node) == input_mask.shape[0]
                        decision = previously_dims_equal and is_dims_equal
                        can_prune_by_dim[producer.id] = PruningAnalysisDecision(
                            decision, PruningAnalysisReason.DIMENSION_MISMATCH
                        )
        # Remove all convolutions with masks
        # that were propagated to output node
        for out_node in self._graph.get_output_nodes():
            for input_mask in get_input_masks(out_node, self._graph):
                if input_mask:
                    for producer in input_mask.mask_producers:
                        can_prune_by_dim[producer.id] = PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)
        # Update decision for nodes which
        # have no closing convolution
        convs_without_closing_conv = {}
        for k, v in can_prune_by_dim.items():
            if v is None:
                convs_without_closing_conv[k] = PruningAnalysisDecision(
                    False, PruningAnalysisReason.CLOSING_CONV_MISSING
                )
        can_prune_by_dim.update(convs_without_closing_conv)

        # Clean nodes masks
        for node in self._graph.get_all_nodes():
            node.attributes["output_mask"] = None

        return can_prune_by_dim

    def _get_can_closing_convs(self, prunable_layers_types) -> Dict:
        retval = set()
        for node in self._graph.get_all_nodes():
            if node.node_type in prunable_layers_types and not (
                is_grouped_conv(node) or is_batched_linear(node, self._graph)
            ):
                retval.add(node.node_id)
        return retval
