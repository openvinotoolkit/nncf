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

import torch

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.torch.graph.operator_metatypes import (
    PTAddMetatype,
    PTAvgPool2dMetatype,
    PTBatchNormMetatype,
    PTCatMetatype,
    PTConv1dMetatype,
    PTConv2dMetatype,
    PTConv3dMetatype,
    PTConvTranspose2dMetatype,
    PTConvTranspose3dMetatype,
    PTDivMetatype,
    PTDropoutMetatype,
    PTELUMetatype,
    PTGELUMetatype,
    PTGroupNormMetatype,
    PTHardTanhMetatype,
    PTHardSwishMetatype,
    PTHardSigmoidMetatype,
    PTInputNoopMetatype,
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
    GroupNormPruningOp,
    ConcatPruningOp,
    ElementwisePruningOp,
    ReshapePruningOp,
    StopMaskForwardPruningOp
)
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
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
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
        """
        Prune node by output_mask (if mask is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
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
    subtypes = [PTHardTanhMetatype, PTTanhMetatype, PTRELUMetatype, PTLeakyRELUMetatype, PTPRELUMetatype, PTELUMetatype,
                PTGELUMetatype, PTSigmoidMetatype, PTSoftmaxMetatype, PTAvgPool2dMetatype, PTMaxPool2dMetatype,
                PTDropoutMetatype, PTSILUMetatype, PTPowerMetatype, PTHardSwishMetatype, PTHardSigmoidMetatype,
                PTNoopMetatype]
    additional_types = ['h_sigmoid', 'h_swish', 'RELU']


@PT_PRUNING_OPERATOR_METATYPES.register('convolution')
class PTConvolutionPruningOp(ConvolutionPruningOp, PTPruner):
    subtypes = [PTConv1dMetatype, PTConv2dMetatype, PTConv3dMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
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

        nncf_logger.info('Pruned Convolution {} by input mask. Old input filters number: {}, new filters number:'
                         ' {}.'.format(node.data['key'], old_num_channels, new_num_channels))

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

        nncf_logger.info('Pruned Convolution {} by pruning mask. Old output filters number: {}, new filters number:'
                         ' {}.'.format(node.data['key'], old_num_clannels, node_module.out_channels))


@PT_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class PTTransposeConvolutionPruningOp(TransposeConvolutionPruningOp, PTPruner):
    subtypes = [PTConvTranspose2dMetatype, PTConvTranspose3dMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
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
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph) -> None:
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

        nncf_logger.info('Pruned BatchNorm {} by input mask. Old num features: {}, new num features:'
                         ' {}.'.format(node.data['key'], old_num_clannels, new_num_channels))


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

        nncf_logger.info('Pruned GroupNorm {} by input mask. Old num features: {}, new num features:'
                         ' {}.'.format(node.data['key'], old_num_clannels, new_num_channels))


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

            nncf_logger.info('Pruned Elementwise {} by input mask. Old num features: {}, new num features:'
                             ' {}.'.format(node.data['key'], old_num_clannels, new_num_channels))


@PT_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class PTStopMaskForwardPruningOp(StopMaskForwardPruningOp, PTPruner):
    subtypes = [PTMeanMetatype, PTMaxMetatype, PTMinMetatype, PTLinearMetatype, PTMatMulMetatype, PTSumMetatype,
                UnknownMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register('reshape')
class PTReshape(ReshapePruningOp):
    subtypes = [PTReshapeMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register('concat')
class PTConcatPruningOp(ConcatPruningOp, PTPruner):
    subtypes = [PTCatMetatype]


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
