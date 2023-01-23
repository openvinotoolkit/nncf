"""
 Copyright (c) 2023 Intel Corporation
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

import torch

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.utils import get_input_masks
from nncf.torch.graph.operator_metatypes import (
    PTAddMetatype,
    PTAvgPool2dMetatype,
    PTBatchNormMetatype,
    PTCatMetatype,
    PTConv1dMetatype,
    PTConv2dMetatype,
    PTConv3dMetatype,
    PTConvTranspose1dMetatype,
    PTConvTranspose2dMetatype,
    PTConvTranspose3dMetatype,
    PTDivMetatype,
    PTDropoutMetatype,
    PTELUMetatype,
    PTRELU6Metatype,
    PTGELUMetatype,
    PTGroupNormMetatype,
    PTLayerNormMetatype,
    PTHardTanhMetatype,
    PTHardSwishMetatype,
    PTHardSigmoidMetatype,
    PTInputNoopMetatype,
    PTInterpolateMetatype,
    PTLinearMetatype,
    PTMatMulMetatype,
    PTMaxMetatype,
    PTMaxPool2dMetatype,
    PTMeanMetatype,
    PTMinMetatype,
    PTMulMetatype,
    PTNoopMetatype,
    PTOutputNoopMetatype,
    PTPowerMetatype,
    PTPRELUMetatype,
    PTLeakyRELUMetatype,
    PTRELUMetatype,
    PTSigmoidMetatype,
    PTSILUMetatype,
    PTSoftmaxMetatype,
    PTSplitMetatype,
    PTSubMetatype,
    PTSumMetatype,
    PTTanhMetatype,
    PTReshapeMetatype
)
from nncf.common.pruning.operations import (
    InputPruningOp,
    OutputPruningOp,
    IdentityMaskForwardPruningOp,
    ConvolutionPruningOp,
    TransposeConvolutionPruningOp,
    BatchNormPruningOp,
    LinearPruningOp,
    GroupNormPruningOp,
    LayerNormPruningOp,
    ConcatPruningOp,
    ElementwisePruningOp,
    ReshapePruningOp,
    StopMaskForwardPruningOp,
    SplitPruningOp
)

from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.logging import nncf_logger
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.tensor_processor import PTNNCFPruningTensorProcessor

PT_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


class PTPruner:
    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
        """
        Prune node by input_masks (if masks is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
        :param graph: Graph of model.
        """

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        """
        Reorder input channels of node by input_masks (if masks is not none and operation support it).
        It's needed to make an equivalent network after sorting filters by importance in the previous layer.

        :param model: NNCF network.
        :param node: Node from NNCF graph that will reorder input channels.
        :param graph: Graph of model.
        """

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
        """
        Prune node by output_mask (if mask is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
        :param graph: Graph of model.
        """

    @classmethod
    def output_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        """
        Reorder output channels of node by output_mask (if masks is not none and operation support it).
        It's needed for performing pruning of filters by simple crop of the last important elements.
        :param model: NNCF network.
        :param node: Node from NNCF graph that will reorder output channels.
        :param graph: Graph of model.
        """


@PT_PRUNING_OPERATOR_METATYPES.register('model_input')
class PTInputPruningOp(InputPruningOp, PTPruner):
    subtypes = [PTInputNoopMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register('model_output')
class PTOutputPruningOp(OutputPruningOp, PTPruner):
    subtypes = [PTOutputNoopMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class PTIdentityMaskForwardPruningOp(IdentityMaskForwardPruningOp, PTPruner):
    subtypes = [PTHardTanhMetatype, PTTanhMetatype, PTRELUMetatype, PTRELU6Metatype, PTLeakyRELUMetatype,
                PTPRELUMetatype, PTELUMetatype, PTGELUMetatype, PTSigmoidMetatype, PTSoftmaxMetatype,
                PTAvgPool2dMetatype, PTMaxPool2dMetatype, PTDropoutMetatype, PTSILUMetatype, PTPowerMetatype,
                PTHardSwishMetatype, PTHardSigmoidMetatype, PTNoopMetatype, PTInterpolateMetatype]
    additional_types = ['h_sigmoid', 'h_swish', 'RELU']


@PT_PRUNING_OPERATOR_METATYPES.register('convolution')
class PTConvolutionPruningOp(ConvolutionPruningOp, PTPruner):
    subtypes = [PTConv1dMetatype, PTConv2dMetatype, PTConv3dMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return
        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        new_num_channels = int(torch.sum(input_mask))

        is_depthwise = is_prunable_depthwise_conv(node)
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

        nncf_logger.debug(
            f'Pruned Convolution {node.data["key"]} by input mask. '
            f'Old input filters number: {old_num_channels}, new filters number: {new_num_channels}.')

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
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

        nncf_logger.debug(
            f'Pruned Convolution {node.data["key"]} by pruning mask. '
            f'Old output filters number: {old_num_clannels}, new filters number: {node_module.out_channels}.')

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        if is_prunable_depthwise_conv(node):
            return
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return
        reorder_indexes = reorder_indexes.tensor
        conv = model.get_containing_module(node.node_name)
        conv.weight.data = torch.index_select(conv.weight.data, 1, reorder_indexes)
        nncf_logger.debug(
            f'Reordered input channels (first 10 reorder indexes {reorder_indexes[:10]}) '
            f'of Convolution: {node.data["key"]} ')

    @classmethod
    def output_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        reorder_indexes = node.data['output_mask']
        if reorder_indexes is None:
            return
        conv = model.get_containing_module(node.node_name)
        reorder_indexes = reorder_indexes.tensor
        conv.weight.data = torch.index_select(conv.weight.data, 0, reorder_indexes)
        if conv.bias is not None:
            conv.bias.data = torch.index_select(conv.bias.data, 0, reorder_indexes)
        nncf_logger.debug(
            f'Reordered output channels (first 10 reorder indexes {reorder_indexes[:10]}) '
            f'of Convolution: {node.data["key"]} ')


@PT_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class PTTransposeConvolutionPruningOp(TransposeConvolutionPruningOp, PTPruner):
    subtypes = [PTConvTranspose1dMetatype, PTConvTranspose2dMetatype, PTConvTranspose3dMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return
        bool_mask = torch.tensor(input_mask, dtype=torch.bool)

        node_module = model.get_containing_module(node.node_name)
        old_num_channels = int(node_module.weight.size(0))

        node_module.in_channels = int(torch.sum(bool_mask))
        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])

        nncf_logger.debug(
            f'Pruned ConvTranspose {node.data["key"]} by input mask. '
            f'Old input filters number: {old_num_channels}, new filters number: {node_module.in_channels}.')

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
        output_mask = node.data['output_mask']
        if output_mask is None:
            return

        bool_mask = torch.tensor(output_mask, dtype=torch.bool)
        new_num_channels = int(torch.sum(bool_mask))

        node_module = model.get_containing_module(node.node_name)
        old_num_channels = int(node_module.weight.size(1))

        in_channels = node_module.weight.size(0)
        broadcasted_mask = bool_mask.repeat(in_channels).view(in_channels, bool_mask.size(0))
        new_weight_shape = list(node_module.weight.shape)
        new_weight_shape[1] = new_num_channels

        node_module.out_channels = new_num_channels
        node_module.weight = torch.nn.Parameter(node_module.weight[broadcasted_mask].view(new_weight_shape))

        if node_module.bias is not None:
            node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

        nncf_logger.debug(
            f'Pruned ConvTranspose {node.data["key"]} by pruning mask. '
            f'Old output filters number: {old_num_channels}, new filters number: {node_module.out_channels}.')


@PT_PRUNING_OPERATOR_METATYPES.register('linear')
class PTLinearPruningOp(LinearPruningOp, PTPruner):
    subtypes = [PTLinearMetatype, PTMatMulMetatype]

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return
        reorder_indexes = reorder_indexes.tensor
        fc = model.get_containing_module(node.node_name)
        fc.weight.data = torch.index_select(fc.weight.data, 1, reorder_indexes)
        nncf_logger.debug(
            'Reordered input channels (first 10 reorder indexes {}) of Linear: {} '.format(reorder_indexes[:10],
                                                                                           node.data['key']))

    @classmethod
    def output_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        reorder_indexes = node.data['output_mask']
        if reorder_indexes is None:
            return
        fc = model.get_containing_module(node.node_name)
        reorder_indexes = reorder_indexes.tensor
        fc.weight.data = torch.index_select(fc.weight.data, 0, reorder_indexes)
        if fc.bias is not None:
            fc.bias.data = torch.index_select(fc.bias.data, 0, reorder_indexes)
        nncf_logger.debug(
            'Reordered output channels (first 10 reorder indexes {}) of Linear: {} '.format(reorder_indexes[:10],
                                                                                            node.data['key']))

@PT_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class PTBatchNormPruningOp(BatchNormPruningOp, PTPruner):
    subtypes = [PTBatchNormMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
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

        nncf_logger.debug(
            f'Pruned BatchNorm {node.data["key"]} by input mask. '
            f'Old num features: {old_num_clannels}, new num features: {new_num_channels}.')

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return

        reorder_indexes = reorder_indexes.tensor
        reorder_indexes = reorder_indexes.int()
        bn = model.get_containing_module(node.node_name)

        bn.weight.data = torch.index_select(bn.weight.data, 0, reorder_indexes)
        bn.bias.data = torch.index_select(bn.bias.data, 0, reorder_indexes)
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, reorder_indexes)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, reorder_indexes)

        nncf_logger.debug(
            f'Reordered channels (first 10 reorder indexes {reorder_indexes[:10]}) of BatchNorm: {node.data["key"]} ')


@PT_PRUNING_OPERATOR_METATYPES.register('group_norm')
class PTGroupNormPruningOp(GroupNormPruningOp, PTPruner):
    subtypes = [PTGroupNormMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
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

        nncf_logger.debug(
            f'Pruned GroupNorm {node.data["key"]} by input mask. '
            f'Old num features: {old_num_clannels}, new num features: {new_num_channels}.')


@PT_PRUNING_OPERATOR_METATYPES.register('layer_norm')
class PTLayerNormPruningOp(LayerNormPruningOp, PTPruner):
    subtypes = [PTLayerNormMetatype]

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return

        reorder_indexes = reorder_indexes.tensor
        ln = model.get_containing_module(node.node_name)
        ln.weight.data = torch.index_select(ln.weight.data, 0, reorder_indexes)
        ln.bias.data = torch.index_select(ln.bias.data, 0, reorder_indexes)

        nncf_logger.debug(
            'Reordered channels (first 10 reorder indexes {}) of LayerNorm: {} '.format(reorder_indexes[:10],
                                                                                        node.data['key']))


@PT_PRUNING_OPERATOR_METATYPES.register('elementwise')
class PTElementwisePruningOp(ElementwisePruningOp, PTPruner):
    subtypes = [PTAddMetatype, PTSubMetatype, PTDivMetatype, PTMulMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
        input_mask = node.data['input_masks'][0]
        if input_mask is None:
            return

        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        node_module = model.get_containing_module(node.node_name)

        if isinstance(node_module, tuple(NNCF_WRAPPED_USER_MODULES_DICT)):
            assert node_module.target_weight_dim_for_compression == 0, \
                "Implemented only for target_weight_dim_for_compression == 0"
            old_num_clannels = int(node_module.weight.size(0))
            new_num_channels = int(torch.sum(input_mask))
            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
            node_module.n_channels = new_num_channels

            nncf_logger.debig(
                f'Pruned Elementwise {node.data["key"]} by input mask. '
                f'Old num features: {old_num_clannels}, new num features: {new_num_channels}.')


@PT_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class PTStopMaskForwardPruningOp(StopMaskForwardPruningOp, PTPruner):
    subtypes = [PTMeanMetatype, PTMaxMetatype, PTMinMetatype, PTSumMetatype,
                UnknownMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register('reshape')
class PTReshape(ReshapePruningOp, PTPruner):
    subtypes = [PTReshapeMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register('concat')
class PTConcatPruningOp(ConcatPruningOp, PTPruner):
    subtypes = [PTCatMetatype]

@PT_PRUNING_OPERATOR_METATYPES.register('chunk')
class PTSplitPruningOp(SplitPruningOp, PTPruner):
    subtypes = [PTSplitMetatype]

class ModelPruner(MaskPropagationAlgorithm):
    def __init__(self, model: NNCFNetwork, graph: NNCFGraph,
                 pruning_operator_metatypes: PruningOperationsMetatypeRegistry):
        super().__init__(graph, pruning_operator_metatypes, PTNNCFPruningTensorProcessor)
        self._model = model

    def apply_mask(self):
        """
        Applying propagated masks for all nodes in topological order:
        1. running input_prune method for this node
        2. running output_prune method for this node
        """
        pruned_node_modules = []
        with torch.no_grad():
            for node in self._graph.topological_sort():
                node_cls = self.get_meta_operation_by_type_name(node.node_type)
                node_module = self._model.get_containing_module(node.node_name)
                if node_module not in pruned_node_modules:
                    node_cls.input_prune(self._model, node, self._graph)
                    node_cls.output_prune(self._model, node, self._graph)
                    pruned_node_modules.append(node_module)
            nncf_logger.info('Finished applying pruning masks.')

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
