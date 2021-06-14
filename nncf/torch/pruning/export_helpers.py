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
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
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
    InputNoopMetatype,
    LinearMetatype,
    MatMulMetatype,
    MaxMetatype,
    MaxPool2dMetatype,
    MeanMetatype,
    MinMetatype,
    MulMetatype,
    OutputNoopMetatype,
    PRELUMetatype,
    RELUMetatype,
    SigmoidMetatype,
    SoftmaxMetatype,
    SubMetatype,
    TanhMetatype,
)
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.pruning.utils import is_depthwise_conv

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

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        """
        Prune node by input_masks (if masks is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
        :param graph: Graph of model.
        """

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        """
        Prune node by output_mask (if mask is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
        :param graph: Graph of model.
        """


@PT_PRUNING_OPERATOR_METATYPES.register('model_input')
class PTInput(PTDefaultMetaOp):
    subtypes = [InputNoopMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['input_masks'] = []
        node.data['output_mask'] = None


@PT_PRUNING_OPERATOR_METATYPES.register('model_output')
class PTOutput(PTDefaultMetaOp):
    subtypes = [OutputNoopMetatype]

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

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return
        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        new_num_channels = int(torch.sum(input_mask))

        is_depthwise = is_depthwise_conv(node)
        node_module = model.get_containing_module(node.node_name)
        old_num_channels = int(node_module.weight.size(1))

        if is_depthwise:
            # In depthwise case prune output channels by input mask, here only fix for new number of input channels
            node_module.groups = new_num_channels
            node_module.in_channels = new_num_channels
            old_num_channels = int(node_module.weight.size(0))
        else:
            out_channels = node_module.weight.size(0)
            broadcasted_mask = bool_mask.repeat(out_channels).view(out_channels, bool_mask.size(0))
            new_weight_shape = list(node_module.weight.shape)
            new_weight_shape[1] = new_num_channels

            node_module.in_channels = new_num_channels
            node_module.weight = torch.nn.Parameter(node_module.weight[broadcasted_mask].view(new_weight_shape))

        nncf_logger.info('Pruned Convolution {} by input mask. Old input filters number: {}, new filters number:'
                         ' {}.'.format(node.data['key'], old_num_channels, new_num_channels))

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        mask = node.data['output_mask']
        if mask is None:
            return

        bool_mask = torch.tensor(mask, dtype=torch.bool)

        node_module = model.get_containing_module(node.node_name)
        old_num_clannels = int(node_module.weight.size(0))

        node_module.out_channels = int(torch.sum(mask))
        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])

        if node_module.bias is not None:
            node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

        nncf_logger.info('Pruned Convolution {} by pruning mask. Old output filters number: {}, new filters number:'
                         ' {}.'.format(node.data['key'], old_num_clannels, node_module.out_channels))


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

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return
        bool_mask = torch.tensor(input_mask, dtype=torch.bool)

        node_module = model.get_containing_module(node.node_name)
        old_num_clannels = int(node_module.weight.size(0))

        node_module.in_channels = int(torch.sum(bool_mask))
        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])

        nncf_logger.info('Pruned ConvTranspose {} by input mask. Old input filters number: {}, new filters number:'
                         ' {}.'.format(node.data['key'], old_num_clannels, node_module.in_channels))

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        output_mask = node.data['output_mask']
        if output_mask is None:
            return

        bool_mask = torch.tensor(output_mask, dtype=torch.bool)
        new_num_channels = int(torch.sum(bool_mask))

        node_module = model.get_containing_module(node.node_name)
        old_num_clannels = int(node_module.weight.size(1))

        in_channels = node_module.weight.size(0)
        broadcasted_mask = bool_mask.repeat(in_channels).view(in_channels, bool_mask.size(0))
        new_weight_shape = list(node_module.weight.shape)
        new_weight_shape[1] = new_num_channels

        node_module.out_channels = new_num_channels
        node_module.weight = torch.nn.Parameter(node_module.weight[broadcasted_mask].view(new_weight_shape))

        if node_module.bias is not None:
            node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

        nncf_logger.info('Pruned ConvTranspose {} by pruning mask. Old output filters number: {}, new filters number:'
                         ' {}.'.format(node.data['key'], old_num_clannels, node_module.out_channels))


@PT_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class PTBatchNorm(PTDefaultMetaOp):
    subtypes = [BatchNormMetatype]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return

        node_module = model.get_containing_module(node.node_name)

        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        old_num_clannels = int(node_module.weight.size(0))
        new_num_channels = int(torch.sum(input_mask))

        node_module.num_features = new_num_channels
        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
        node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])
        node_module.running_mean = torch.nn.Parameter(node_module.running_mean[bool_mask], requires_grad=False)
        node_module.running_var = torch.nn.Parameter(node_module.running_var[bool_mask], requires_grad=False)

        nncf_logger.info('Pruned BatchNorm {} by input mask. Old num features: {}, new num features:'
                         ' {}.'.format(node.data['key'], old_num_clannels, new_num_channels))


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

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return

        node_module = model.get_containing_module(node.node_name)

        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        old_num_clannels = int(node_module.weight.size(0))
        new_num_channels = int(torch.sum(input_mask))

        node_module.num_channels = new_num_channels
        node_module.num_groups = new_num_channels

        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
        node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

        nncf_logger.info('Pruned GroupNorm {} by input mask. Old num features: {}, new num features:'
                         ' {}.'.format(node.data['key'], old_num_clannels, new_num_channels))


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
        input_edges_desc = list(input_edges.values())
        previous_nodes = [graph.get_node_by_key(edge[0]) for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        if all(mask is None for mask in input_masks):
            return None

        device = [m for m in input_masks if m is not None][0].device

        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                mask = torch.ones(input_edges_desc[i][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR][1], device=device)
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

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return

        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        node_module = model.get_containing_module(node.node_name)

        if isinstance(node_module, tuple(NNCF_WRAPPED_USER_MODULES_DICT)):
            assert node_module.target_weight_dim_for_compression == 0,\
                "Implemented only for target_weight_dim_for_compression == 0"
            old_num_clannels = int(node_module.weight.size(0))
            new_num_channels = int(torch.sum(input_mask))
            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
            node_module.n_channels = new_num_channels

            nncf_logger.info('Pruned Elementwise {} by input mask. Old num features: {}, new num features:'
                             ' {}.'.format(node.data['key'], old_num_clannels, new_num_channels))


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


class ModelPruner(MaskPropagationAlgorithm):
    def __init__(self, model: NNCFNetwork, graph: NNCFGraph,
                 pruning_operator_metatypes: PruningOperationsMetatypeRegistry):
        super().__init__(graph, pruning_operator_metatypes)
        self._model = model

    def apply_mask(self):
        """
        Applying propagated masks for all nodes in topological order:
        1. running input_prune method for this node
        2. running output_prune method for this node
        """
        pruned_node_modules = list()
        with torch.no_grad():
            for node in self._graph.topological_sort():
                node_cls = self.get_meta_operation_by_type_name(node.node_type)
                node_module = self._model.get_containing_module(node.node_name)
                if node_module not in pruned_node_modules:
                    node_cls.input_prune(self._model, node, self._graph)
                    node_cls.output_prune(self._model, node, self._graph)
                    pruned_node_modules.append(node_module)
            nncf_logger.info('Finished mask applying step')

    def prune_model(self):
        """
        Model pruner work in two stages:
        1. Mask propagation: propagate pruning masks through the graph.
        2. Applying calculated masks
        """
        nncf_logger.info('Start pruning model')
        self.mask_propagation()
        self.apply_mask()
        nncf_logger.info('Finished pruning model')
