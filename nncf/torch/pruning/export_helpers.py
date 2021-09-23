"""
 Copyright (c) 2020 Intel Corporation
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
from typing import Union
from typing import List

import torch

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.export_helpers import DefaultMetaOp
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.mask_propagation import identity_mask_propagation
from nncf.common.pruning.mask_propagation import get_input_masks
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.torch.graph.operator_metatypes import (
    AddMetatype,
    AvgPool2dMetatype,
    BatchNormMetatype,
    CatMetatype,
    Conv1dMetatype,
    Conv2dMetatype,
    Conv3dMetatype,
    ConvTranspose2dMetatype,
    ConvTranspose3dMetatype,
    DivMetatype,
    DropoutMetatype,
    ELUMetatype,
    GELUMetatype,
    GroupNormMetatype,
    HardTanhMetatype,
    PTInputNoopMetatype,
    LinearMetatype,
    MatMulMetatype,
    MaxMetatype,
    MaxPool2dMetatype,
    MeanMetatype,
    MinMetatype,
    MulMetatype,
    PTOutputNoopMetatype,
    PRELUMetatype,
    RELUMetatype,
    SigmoidMetatype,
    SoftmaxMetatype,
    SubMetatype,
    TanhMetatype,
)
from nncf.common.pruning.utils import is_depthwise_conv

PT_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


class PTDefaultMetaOp(DefaultMetaOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        """
        Propagate mask through a node using masks of all inputs and pruning mask of current node (if any).
        Should set the following attributes:
            input_masks - list of masks of input nodes (None if there is no mask in some input);
            output_mask - resulting mask of node operation.

        :param node: Node from NNCF graph to propagate mask through it.
        :param graph: Graph of model to prune.
        """
        raise NotImplementedError


@PT_PRUNING_OPERATOR_METATYPES.register('model_input')
class PTInput(PTDefaultMetaOp):
    subtypes = [PTInputNoopMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['input_masks'] = []
        node.data['output_mask'] = None


@PT_PRUNING_OPERATOR_METATYPES.register('model_output')
class PTOutput(PTDefaultMetaOp):
    subtypes = [PTOutputNoopMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['input_masks'] = []
        node.data['output_mask'] = None


@PT_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class PTIdentityMaskForwardOps(PTDefaultMetaOp):
    subtypes = [HardTanhMetatype, TanhMetatype, RELUMetatype, PRELUMetatype, ELUMetatype, GELUMetatype, SigmoidMetatype,
                SoftmaxMetatype, AvgPool2dMetatype, MaxPool2dMetatype, DropoutMetatype]
    additional_types = ['h_sigmoid', 'h_swish', 'RELU']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


@PT_PRUNING_OPERATOR_METATYPES.register('convolution')
class PTConvolution(PTDefaultMetaOp):
    subtypes = [Conv1dMetatype, Conv2dMetatype, Conv3dMetatype]

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

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = output_mask


@PT_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class PTTransposeConvolution(PTDefaultMetaOp):
    subtypes = [ConvTranspose2dMetatype, ConvTranspose3dMetatype]

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

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = output_mask


@PT_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class PTBatchNorm(PTDefaultMetaOp):
    subtypes = [BatchNormMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


@PT_PRUNING_OPERATOR_METATYPES.register('group_norm')
class GroupNorm(PTDefaultMetaOp):
    subtypes = [GroupNormMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


@PT_PRUNING_OPERATOR_METATYPES.register('concat')
class PTConcat(PTDefaultMetaOp):
    subtypes = [CatMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def check_concat(cls, node: NNCFNode, graph: NNCFGraph) -> bool:
        """
        Return whether all input sources of node is convolutions or not.

        :param node: Node to determine it's sources.
        :param graph: NNCF graph to work with.
        :return: True If all input sources of node is convolutions.
        """

        for input_node in graph.get_previous_nodes(node):
            # If input has mask ->  it went from convolution (source of this node is a convolution)
            if input_node.data.get('output_mask', None) is None:
                continue

            source_nodes = get_sources_of_node(input_node, graph, PTConvolution.get_all_op_aliases() +
                                               PTStopMaskForwardOps.get_all_op_aliases() +
                                               PTInput.get_all_op_aliases())
            sources_types = [node.node_type for node in source_nodes]
            if any(t in sources_types for t in PTStopMaskForwardOps.get_all_op_aliases()):
                return False
        return True

    @classmethod
    def fill_input_masks(cls, node: NNCFNode, graph: NNCFGraph) -> Union[List[torch.Tensor], None]:
        """
        Fill input masks with all None replaced by identity masks.
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

        device = [m for m in input_masks if m is not None][0].device

        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                mask = torch.ones(input_edges[i].tensor_shape[1], device=device)
            filled_input_masks.append(mask)
        return filled_input_masks

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = None
        output_mask = None

        if cls.check_concat(node, graph):
            input_masks = cls.fill_input_masks(node, graph)
            if input_masks:
                output_mask = torch.cat(input_masks)

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = output_mask


@PT_PRUNING_OPERATOR_METATYPES.register('elementwise')
class PTElementwise(PTDefaultMetaOp):
    subtypes = [AddMetatype, SubMetatype, DivMetatype, MulMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        node.data['input_masks'] = input_masks
        if input_masks[0] is not None:
            assert all(torch.allclose(input_masks[0], mask) for mask in input_masks)
        node.data['output_mask'] = input_masks[0]


@PT_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class PTStopMaskForwardOps(PTDefaultMetaOp):
    subtypes = [MeanMetatype, MaxMetatype, MinMetatype, LinearMetatype, MatMulMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = None
