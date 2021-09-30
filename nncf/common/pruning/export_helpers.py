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

from typing import Union

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.utils import is_depthwise_conv
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.pruning.mask_propagation import identity_mask_propagation
from nncf.common.pruning.mask_propagation import get_input_masks


class DefaultMetaOp:
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


class OpInput(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None


class OpOutput(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None


class OpIdentityMaskForwardOps(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class OpConvolution(DefaultMetaOp):
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


class OpTransposeConvolution(DefaultMetaOp):
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


class OpBatchNorm(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class OpGroupNorm(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


class OpConcat(DefaultMetaOp):
    ConvolutionOp = None # type: OpConvolution
    StopMaskForwardOp = None # type: OpStopMaskForwardOps
    InputOp = None # type: OpInput

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
            if input_node.data.get('output_mask', None) is None:
                continue

            source_nodes = get_sources_of_node(input_node, graph, cls.ConvolutionOp.get_all_op_aliases() +
                                               cls.StopMaskForwardOp.get_all_op_aliases() +
                                               cls.InputOp.get_all_op_aliases())
            sources_types = [node.node_type for node in source_nodes]
            if any(t in sources_types for t in cls.StopMaskForwardOp.get_all_op_aliases()):
                return False
        return True

    @classmethod
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph) -> Union[np.array, None]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources
        :param graph: NNCF graph to work with
        :return: Output mask
        """
        input_edges = graph.get_input_edges(node)
        previous_nodes = [edge.from_node for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        if all(mask is None for mask in input_masks):
            return None


        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                mask = np.ones(input_edges[i].tensor_shape[-1])
            filled_input_masks.append(mask)
        result_mask = np.concatenate(filled_input_masks, 0)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        result_mask = None

        if cls.check_concat(node, graph):
            result_mask = cls.generate_output_mask(node, graph)

        node.data['output_mask'] = result_mask


class OpElementwise(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        if input_masks[0] is not None:
            for input_mask in input_masks[1:]:
                np.testing.assert_allclose(input_masks[0], input_mask)
        node.data['output_mask'] = input_masks[0]


class OpReshape(DefaultMetaOp):
    @staticmethod
    def _is_flatten(node: NNCFNode):
        return sum([dim for dim in node.layer_attributes.output_shape if dim]) == 1

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        if node.layer_attributes is None:
            return False
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        if cls.accept_pruned_input(node):
            if cls._is_flatten(node):
                OpFlatten.mask_propagation(node, graph)
            else:
                identity_mask_propagation(node, graph)
        else:
            node.data['output_mask'] = None


class OpFlatten(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        if node.layer_attributes is not None:
            return True
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        output_mask = None
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is not None and node.layer_attributes is not None:
            flatten_channels = int(np.prod(node.layer_attributes.input_shape))
            mask_len = input_mask.shape[0]
            assert flatten_channels % mask_len == 0
            output_mask = np.repeat(input_mask, repeats=flatten_channels // mask_len)
        node.data['output_mask'] = output_mask


class OpStopMaskForwardOps(DefaultMetaOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None
