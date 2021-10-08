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

import numpy as np

from typing import Union, List

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.utils import is_depthwise_conv
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import identity_mask_propagation
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.tensor import NNCFTensor


class BasePruningOp:
    """
    Determines meta operations which aggregate operations having common
    properties of interaction with pruning masks
    """

    subtypes = []
    additional_types = []

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        """
        :return: accept_pruned_input - can this operation work with pruned input or not
        """
        raise NotImplementedError

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        """
        Propagates the pruning mask through a node using pruning masks of all inputs and the current node (if any).

        :param node: The graph node to propagate mask through it
        :param graph: The model graph to prune
        """
        raise NotImplementedError

    @classmethod
    def get_all_op_aliases(cls):
        """
        :return: list of all aliases of types in metatype
        """
        op_types = []
        for subtype in cls.subtypes:
            op_types.extend(subtype.get_all_aliases())
        op_types = list(set(op_types)) + cls.additional_types
        return op_types


class InputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None


class OutputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None


class IdentityMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class ConvolutionPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        accept_pruned_input = True
        if is_grouped_conv(node):
            if not is_depthwise_conv(node):
                accept_pruned_input = False
        return accept_pruned_input

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        if is_grouped_conv(node):
            output_mask = None
            if is_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class TransposeConvolutionPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        accept_pruned_input = True
        if is_grouped_conv(node):
            if not is_depthwise_conv(node):
                accept_pruned_input = False
        return accept_pruned_input

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        # In case of group convs we can't prune by output filters
        if is_grouped_conv(node):
            output_mask = None
            if is_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class BatchNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class ConcatPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph) -> Union[List[np.array], None]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources.
        :param graph: NNCF graph to work with.
        :return: Filled input masks.
        """
        input_edges = graph.get_input_edges(node)
        previous_nodes = [edge.from_node for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        not_empty_masks = [mask for mask in input_masks if mask is not None] # type: List[NNCFTensor]
        if not not_empty_masks:
            return None

        first_non_empty_mask = not_empty_masks[0]
        tensor_processor = first_non_empty_mask.mask_processor
        device = first_non_empty_mask.device
        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                concat_axis = node.layer_attributes.axis
                concat_dim = input_edges[i].tensor_shape[concat_axis]
                mask = tensor_processor.ones(concat_dim, device)
            filled_input_masks.append(mask)
        result_mask = tensor_processor.concatenate(filled_input_masks, 0)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        result_mask = cls.generate_output_mask(node, graph)
        node.data['output_mask'] = result_mask


class ElementwisePruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        node.data['input_masks'] = input_masks # type: List[NNCFTensor]
        if input_masks[0] is not None:
            input_masks[0].mask_processor.check_all_close(input_masks)
        node.data['output_mask'] = input_masks[0]


class StopMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None
