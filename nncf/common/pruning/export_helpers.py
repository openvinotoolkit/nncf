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
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.pruning.mask_propagation import identity_mask_propagation
from nncf.common.pruning.mask_propagation import get_input_masks
from nncf.common.pruning.default_pruning_op import DefaultPruningOp


class InputPruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None


class OutputPruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None


class IdentityMaskForwardPruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class ConvolutionPruningOp(DefaultPruningOp):
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


class TransposeConvolutionPruningOp(DefaultPruningOp):
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


class BatchNormPruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class ConcatPruningOp(DefaultPruningOp):
    ConvolutionOp = None # type: ConvolutionPruningOp
    StopMaskForwardOp = None # type: StopMaskForwardPruningOp
    InputOp = None # type: InputPruningOp

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def check_concat(cls, node: NNCFNode, graph: NNCFGraph) -> bool:
        """
        Return whether all input sources of node is convolutions or not.

        :param node: Node to determine it's sources
        :param graph: NNCF graph to work with
        :return: True if all input sources of node is convolutions
        """

        for input_node in graph.get_previous_nodes(node):
            # If input has mask ->  it went from convolution (source of this node is a convolution)
            if input_node.data.get('output_mask', None) is not None:
                continue

            source_nodes = get_sources_of_node(input_node, graph, cls.ConvolutionOp.get_all_op_aliases() +
                                               cls.StopMaskForwardOp.get_all_op_aliases() +
                                               cls.InputOp.get_all_op_aliases())
            sources_types = [node.node_type for node in source_nodes] + [input_node.node_type]
            if any(t in sources_types for t in cls.StopMaskForwardOp.get_all_op_aliases()):
                return False
        return True

    @classmethod
    def _get_unit_mask(cls, dim, device):
        return np.ones(dim)

    @classmethod
    def _get_masks_device(cls, input_masks):
        return None

    @classmethod
    def _concat_masks(cls, filled_input_masks):
        return np.concatenate(filled_input_masks, 0)

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

        if all(mask is None for mask in input_masks):
            return None

        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                concat_axis = node.layer_attributes.axis
                concat_dim = input_edges[i].tensor_shape[concat_axis]
                device = cls._get_masks_device(input_masks)
                mask = cls._get_unit_mask(concat_dim, device)
            filled_input_masks.append(mask)
        result_mask = cls._concat_masks(filled_input_masks)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        result_mask = None

        if cls.check_concat(node, graph):
            result_mask = cls.generate_output_mask(node, graph)

        node.data['output_mask'] = result_mask


class ElementwisePruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def _assert_input_masks_close(cls, input_masks):
        for input_mask in input_masks[1:]:
            np.testing.assert_allclose(input_masks[0], input_mask)

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        node.data['input_masks'] = input_masks
        if input_masks[0] is not None:
            cls._assert_input_masks_close(input_masks)
        node.data['output_mask'] = input_masks[0]


class StopMaskForwardPruningOp(DefaultPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None
