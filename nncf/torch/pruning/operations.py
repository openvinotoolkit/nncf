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

from enum import Enum
from enum import auto

import torch

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.logging import nncf_logger
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.operations import BatchNormPruningOp
from nncf.common.pruning.operations import ConcatPruningOp
from nncf.common.pruning.operations import ConvolutionPruningOp
from nncf.common.pruning.operations import ElementwisePruningOp
from nncf.common.pruning.operations import GroupNormPruningOp
from nncf.common.pruning.operations import IdentityMaskForwardPruningOp
from nncf.common.pruning.operations import InputPruningOp
from nncf.common.pruning.operations import LayerNormPruningOp
from nncf.common.pruning.operations import LinearPruningOp
from nncf.common.pruning.operations import OutputPruningOp
from nncf.common.pruning.operations import PadPruningOp
from nncf.common.pruning.operations import ReshapePruningOp
from nncf.common.pruning.operations import SplitPruningOp
from nncf.common.pruning.operations import StopMaskForwardPruningOp
from nncf.common.pruning.operations import TransposeConvolutionPruningOp
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.torch.graph.operator_metatypes import PTAdaptiveMaxPool2dMetatype
from nncf.torch.graph.operator_metatypes import PTAdaptiveMaxPool3dMetatype
from nncf.torch.graph.operator_metatypes import PTAddMetatype
from nncf.torch.graph.operator_metatypes import PTAvgPool2dMetatype
from nncf.torch.graph.operator_metatypes import PTAvgPool3dMetatype
from nncf.torch.graph.operator_metatypes import PTBatchNormMetatype
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTConv1dMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTConv3dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose1dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import PTDivMetatype
from nncf.torch.graph.operator_metatypes import PTDropoutMetatype
from nncf.torch.graph.operator_metatypes import PTELUMetatype
from nncf.torch.graph.operator_metatypes import PTGELUMetatype
from nncf.torch.graph.operator_metatypes import PTGroupNormMetatype
from nncf.torch.graph.operator_metatypes import PTHardSigmoidMetatype
from nncf.torch.graph.operator_metatypes import PTHardSwishMetatype
from nncf.torch.graph.operator_metatypes import PTHardTanhMetatype
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTInterpolateMetatype
from nncf.torch.graph.operator_metatypes import PTLayerNormMetatype
from nncf.torch.graph.operator_metatypes import PTLeakyRELUMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTMatMulMetatype
from nncf.torch.graph.operator_metatypes import PTMaxMetatype
from nncf.torch.graph.operator_metatypes import PTMaxPool2dMetatype
from nncf.torch.graph.operator_metatypes import PTMaxPool3dMetatype
from nncf.torch.graph.operator_metatypes import PTMeanMetatype
from nncf.torch.graph.operator_metatypes import PTMinMetatype
from nncf.torch.graph.operator_metatypes import PTMulMetatype
from nncf.torch.graph.operator_metatypes import PTNoopMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTPadMetatype
from nncf.torch.graph.operator_metatypes import PTPowerMetatype
from nncf.torch.graph.operator_metatypes import PTPRELUMetatype
from nncf.torch.graph.operator_metatypes import PTRELU6Metatype
from nncf.torch.graph.operator_metatypes import PTRELUMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PTSigmoidMetatype
from nncf.torch.graph.operator_metatypes import PTSILUMetatype
from nncf.torch.graph.operator_metatypes import PTSoftmaxMetatype
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.graph.operator_metatypes import PTSqueezeMetatype
from nncf.torch.graph.operator_metatypes import PTSubMetatype
from nncf.torch.graph.operator_metatypes import PTSumMetatype
from nncf.torch.graph.operator_metatypes import PTTanhMetatype
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.torch.pruning.tensor_processor import PTNNCFPruningTensorProcessor
from nncf.torch.tensor import PTNNCFTensor

PT_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


class PrunType(Enum):
    CUT_WEIGHTS = auto()
    FILL_ZEROS = auto()


class PTPruner:
    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        """
        Prune node by input_masks (if masks is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
        :param graph: Graph of model.
        :param prun_type: Type of pruning.
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
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        """
        Prune node by output_mask (if mask is not none and operation support it).

        :param model: NNCF network.
        :param node: Node from NNCF graph that will be prune.
        :param graph: Graph of model.
        :param prun_type: Type of pruning.
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


@PT_PRUNING_OPERATOR_METATYPES.register("model_input")
class PTInputPruningOp(InputPruningOp, PTPruner):
    subtypes = [PTInputNoopMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register("model_output")
class PTOutputPruningOp(OutputPruningOp, PTPruner):
    subtypes = [PTOutputNoopMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register("identity_mask_propagation")
class PTIdentityMaskForwardPruningOp(IdentityMaskForwardPruningOp, PTPruner):
    subtypes = [
        PTHardTanhMetatype,
        PTTanhMetatype,
        PTRELUMetatype,
        PTRELU6Metatype,
        PTLeakyRELUMetatype,
        PTPRELUMetatype,
        PTELUMetatype,
        PTGELUMetatype,
        PTSigmoidMetatype,
        PTSoftmaxMetatype,
        PTAdaptiveMaxPool2dMetatype,
        PTAvgPool2dMetatype,
        PTMaxPool2dMetatype,
        PTAdaptiveMaxPool3dMetatype,
        PTAvgPool3dMetatype,
        PTMaxPool3dMetatype,
        PTMeanMetatype,
        PTDropoutMetatype,
        PTSILUMetatype,
        PTPowerMetatype,
        PTHardSwishMetatype,
        PTHardSigmoidMetatype,
        PTNoopMetatype,
        PTInterpolateMetatype,
    ]
    additional_types = ["h_sigmoid", "h_swish", "RELU"]


@PT_PRUNING_OPERATOR_METATYPES.register("convolution")
class PTConvolutionPruningOp(ConvolutionPruningOp, PTPruner):
    subtypes = [PTConv1dMetatype, PTConv2dMetatype, PTConv3dMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        is_depthwise = is_prunable_depthwise_conv(node)
        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(input_mask, dtype=torch.bool)
            new_num_channels = int(torch.sum(input_mask))
            old_num_channels = int(node_module.weight.size(0))
            if is_depthwise:
                # In depthwise case prune output channels by input mask, here only fix for new number of input channels
                node_module.groups = new_num_channels
                node_module.in_channels = new_num_channels
            else:
                out_channels = node_module.weight.size(0)
                broadcasted_mask = bool_mask.repeat(out_channels).view(out_channels, bool_mask.size(0))
                new_weight_shape = list(node_module.weight.shape)
                new_weight_shape[1] = new_num_channels

                node_module.in_channels = new_num_channels
                node_module.weight = torch.nn.Parameter(node_module.weight[broadcasted_mask].view(new_weight_shape))
                nncf_logger.debug(
                    f"Pruned Convolution {node.node_key} by input mask. "
                    f"Old input filters number: {old_num_channels}, new filters number: {new_num_channels}."
                )
        else:
            if not is_depthwise:
                node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight, dim=1))

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        mask = node.attributes["output_mask"]
        if mask is None:
            return

        if isinstance(mask, PTNNCFTensor):
            mask = mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)
        if prun_type == PrunType.CUT_WEIGHTS:
            old_num_channels = int(node_module.weight.size(0))
            bool_mask = torch.tensor(mask, dtype=torch.bool)

            node_module.out_channels = int(torch.sum(mask))
            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])

            if node_module.bias is not None:
                node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

            nncf_logger.debug(
                f"Pruned Convolution {node.node_key} by pruning mask. "
                f"Old output filters number: {old_num_channels}, new filters number: {node_module.out_channels}."
            )
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(mask, node_module.weight))
            if node_module.bias is not None:
                node_module.bias = torch.nn.Parameter(apply_filter_binary_mask(mask, node_module.bias))

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        if is_prunable_depthwise_conv(node):
            return
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return
        reorder_indexes = reorder_indexes.tensor
        conv = model.nncf.get_containing_module(node.node_name)
        conv.weight.data = torch.index_select(conv.weight.data, 1, reorder_indexes)
        nncf_logger.debug(
            f"Reordered input channels (first 10 reorder indexes {reorder_indexes[:10]}) "
            f"of Convolution: {node.node_key} "
        )

    @classmethod
    def output_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        reorder_indexes = node.attributes["output_mask"]
        if reorder_indexes is None:
            return
        conv = model.nncf.get_containing_module(node.node_name)
        reorder_indexes = reorder_indexes.tensor
        conv.weight.data = torch.index_select(conv.weight.data, 0, reorder_indexes)
        if conv.bias is not None:
            conv.bias.data = torch.index_select(conv.bias.data, 0, reorder_indexes)
        nncf_logger.debug(
            f"Reordered output channels (first 10 reorder indexes {reorder_indexes[:10]}) "
            f"of Convolution: {node.node_key} "
        )


@PT_PRUNING_OPERATOR_METATYPES.register("transpose_convolution")
class PTTransposeConvolutionPruningOp(TransposeConvolutionPruningOp, PTPruner):
    subtypes = [PTConvTranspose1dMetatype, PTConvTranspose2dMetatype, PTConvTranspose3dMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)
        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(input_mask, dtype=torch.bool)
            old_num_channels = int(node_module.weight.size(0))

            node_module.in_channels = int(torch.sum(bool_mask))
            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])

            nncf_logger.debug(
                f"Pruned ConvTranspose {node.node_key} by input mask. "
                f"Old input filters number: {old_num_channels}, new filters number: {node_module.in_channels}."
            )
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight))

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        output_mask = node.attributes["output_mask"]
        if output_mask is None:
            return

        if isinstance(output_mask, PTNNCFTensor):
            output_mask = output_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(output_mask, dtype=torch.bool)
            new_num_channels = int(torch.sum(bool_mask))

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
                f"Pruned ConvTranspose {node.node_key} by pruning mask. "
                f"Old output filters number: {old_num_channels}, new filters number: {node_module.out_channels}."
            )
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(output_mask, node_module.weight, dim=1))
            if node_module.bias is not None:
                node_module.bias = torch.nn.Parameter(apply_filter_binary_mask(output_mask, node_module.bias))


@PT_PRUNING_OPERATOR_METATYPES.register("linear")
class PTLinearPruningOp(LinearPruningOp, PTPruner):
    subtypes = [PTLinearMetatype, PTMatMulMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(input_mask, dtype=torch.bool)
            in_features = node_module.in_features
            out_features = node_module.out_features
            new_in_features = sum(bool_mask)
            node_module.in_features = new_in_features
            broadcasted_mask = bool_mask.repeat(out_features).view(out_features, bool_mask.size(0))
            new_weight_shape = list(node_module.weight.shape)
            new_weight_shape[1] = new_in_features

            node_module.weight = torch.nn.Parameter(node_module.weight[broadcasted_mask].view(new_weight_shape))
            nncf_logger.debug(
                f"Pruned Linear {node.node_key} by pruning mask. "
                f"Old input filters number: {in_features}, new filters number: {node_module.in_features}."
            )
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight, dim=1))

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return
        reorder_indexes = reorder_indexes.tensor
        fc = model.nncf.get_containing_module(node.node_name)
        fc.weight.data = torch.index_select(fc.weight.data, 1, reorder_indexes)
        nncf_logger.debug(
            f"Reordered input channels (first 10 reorder indexes {reorder_indexes[:10]}) of Linear: {node.node_key}"
        )

    @classmethod
    def output_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        output_mask = node.attributes["output_mask"]
        if output_mask is None:
            return

        if isinstance(output_mask, PTNNCFTensor):
            output_mask = output_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(output_mask, dtype=torch.bool)
            out_features = node_module.out_features
            new_out_features = sum(bool_mask)
            node_module.out_features = new_out_features
            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
            nncf_logger.debug(
                f"Pruned Linear {node.node_key} by pruning mask. "
                f"Old output filters number: {out_features}, new filters number: {node_module.out_features}."
            )
            if node_module.bias is not None:
                node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(output_mask, node_module.weight))
            if node_module.bias is not None:
                node_module.bias = torch.nn.Parameter(apply_filter_binary_mask(output_mask, node_module.bias))

    @classmethod
    def output_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        reorder_indexes = node.attributes["output_mask"]
        if reorder_indexes is None:
            return
        fc = model.nncf.get_containing_module(node.node_name)
        reorder_indexes = reorder_indexes.tensor
        fc.weight.data = torch.index_select(fc.weight.data, 0, reorder_indexes)
        if fc.bias is not None:
            fc.bias.data = torch.index_select(fc.bias.data, 0, reorder_indexes)
        nncf_logger.debug(
            f"Reordered output channels (first 10 reorder indexes {reorder_indexes[:10]}) of Linear: {node.node_key}"
        )


@PT_PRUNING_OPERATOR_METATYPES.register("batch_norm")
class PTBatchNormPruningOp(BatchNormPruningOp, PTPruner):
    subtypes = [PTBatchNormMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(input_mask, dtype=torch.bool)

            old_num_channels = int(node_module.weight.size(0))
            new_num_channels = int(torch.sum(input_mask))

            node_module.num_features = new_num_channels
            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
            node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])
            node_module.running_mean = torch.nn.Parameter(node_module.running_mean[bool_mask], requires_grad=False)
            node_module.running_var = torch.nn.Parameter(node_module.running_var[bool_mask], requires_grad=False)

            nncf_logger.debug(
                f"Pruned BatchNorm {node.node_key} by input mask. "
                f"Old num features: {old_num_channels}, new num features: {new_num_channels}."
            )
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight))
            node_module.bias = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.bias))
            node_module.running_mean = torch.nn.Parameter(
                apply_filter_binary_mask(input_mask, node_module.running_mean), requires_grad=False
            )
            node_module.running_var = torch.nn.Parameter(
                apply_filter_binary_mask(input_mask, node_module.running_var), requires_grad=False
            )

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return

        reorder_indexes = reorder_indexes.tensor
        reorder_indexes = reorder_indexes.int()
        bn = model.nncf.get_containing_module(node.node_name)

        bn.weight.data = torch.index_select(bn.weight.data, 0, reorder_indexes)
        bn.bias.data = torch.index_select(bn.bias.data, 0, reorder_indexes)
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, reorder_indexes)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, reorder_indexes)

        nncf_logger.debug(
            f"Reordered channels (first 10 reorder indexes {reorder_indexes[:10]}) of BatchNorm: {node.node_key} "
        )


@PT_PRUNING_OPERATOR_METATYPES.register("group_norm")
class PTGroupNormPruningOp(GroupNormPruningOp, PTPruner):
    subtypes = [PTGroupNormMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            bool_mask = torch.tensor(input_mask, dtype=torch.bool)
            old_num_channels = int(node_module.weight.size(0))
            new_num_channels = int(torch.sum(input_mask))

            node_module.num_channels = new_num_channels
            node_module.num_groups = new_num_channels

            node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
            node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

            nncf_logger.debug(
                f"Pruned GroupNorm {node.node_key} by input mask. "
                f"Old num features: {old_num_channels}, new num features: {new_num_channels}."
            )
        else:
            node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight))
            node_module.bias = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.bias))


@PT_PRUNING_OPERATOR_METATYPES.register("layer_norm")
class PTLayerNormPruningOp(LayerNormPruningOp, PTPruner):
    subtypes = [PTLayerNormMetatype]

    @classmethod
    def input_reorder(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        reorder_indexes = input_masks[0]
        if reorder_indexes is None:
            return

        reorder_indexes = reorder_indexes.tensor
        ln = model.nncf.get_containing_module(node.node_name)
        ln.weight.data = torch.index_select(ln.weight.data, 0, reorder_indexes)
        ln.bias.data = torch.index_select(ln.bias.data, 0, reorder_indexes)

        nncf_logger.debug(
            "Reordered channels (first 10 reorder indexes {}) of LayerNorm: {} ".format(
                reorder_indexes[:10], node.node_key
            )
        )

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if prun_type == PrunType.CUT_WEIGHTS:
            raise nncf.InternalError("LayerNorm does not support pruning by cutting channels")

        node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight))
        node_module.bias = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.bias))


@PT_PRUNING_OPERATOR_METATYPES.register("elementwise")
class PTElementwisePruningOp(ElementwisePruningOp, PTPruner):
    subtypes = [PTAddMetatype, PTSubMetatype, PTDivMetatype, PTMulMetatype]

    @classmethod
    def input_prune(cls, model: NNCFNetwork, node: NNCFNode, graph: NNCFGraph, prun_type: PrunType) -> None:
        input_mask = get_input_masks(node, graph)[0]
        if input_mask is None:
            return

        if isinstance(input_mask, PTNNCFTensor):
            input_mask = input_mask.tensor

        node_module = model.nncf.get_containing_module(node.node_name)

        if isinstance(node_module, tuple(NNCF_WRAPPED_USER_MODULES_DICT)):
            assert (
                node_module.target_weight_dim_for_compression == 0
            ), "Implemented only for target_weight_dim_for_compression == 0"
            if prun_type == PrunType.CUT_WEIGHTS:
                bool_mask = torch.tensor(input_mask, dtype=torch.bool)
                old_num_channels = int(node_module.weight.size(0))
                new_num_channels = int(torch.sum(input_mask))
                node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
                node_module.n_channels = new_num_channels

                nncf_logger.debug(
                    f"Pruned Elementwise {node.node_key} by input mask. "
                    f"Old num features: {old_num_channels}, new num features: {new_num_channels}."
                )
            else:
                node_module.weight = torch.nn.Parameter(apply_filter_binary_mask(input_mask, node_module.weight))


@PT_PRUNING_OPERATOR_METATYPES.register("stop_propagation_ops")
class PTStopMaskForwardPruningOp(StopMaskForwardPruningOp, PTPruner):
    subtypes = [PTMaxMetatype, PTMinMetatype, PTSumMetatype, UnknownMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register("reshape")
class PTReshape(ReshapePruningOp, PTPruner):
    subtypes = [PTReshapeMetatype, PTSqueezeMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register("concat")
class PTConcatPruningOp(ConcatPruningOp, PTPruner):
    subtypes = [PTCatMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register("chunk")
class PTSplitPruningOp(SplitPruningOp, PTPruner):
    subtypes = [PTSplitMetatype]


@PT_PRUNING_OPERATOR_METATYPES.register("pad")
class PTPadPruningOp(PadPruningOp, PTPruner):
    subtypes = [PTPadMetatype]


class ModelPruner(MaskPropagationAlgorithm):
    def __init__(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
        prun_type: PrunType = PrunType.FILL_ZEROS,
    ):
        super().__init__(graph, pruning_operator_metatypes, PTNNCFPruningTensorProcessor)
        self._model = model
        self._prun_type = prun_type

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
                node_module = self._model.nncf.get_containing_module(node.node_name)
                if node_module not in pruned_node_modules:
                    node_cls.input_prune(self._model, node, self._graph, self._prun_type)
                    node_cls.output_prune(self._model, node, self._graph, self._prun_type)
                    pruned_node_modules.append(node_module)
            nncf_logger.info("Finished applying pruning masks.")

    def remove_filter_pruning_operations(self) -> None:
        """
        Remove all filter pruning operation in the model.

        :param model: Target model.
        """
        for node in self._model.nncf.get_original_graph().get_all_nodes():
            if node.node_type in ["nncf_model_input", "nncf_model_output"]:
                continue

            nncf_module = self._model.nncf.get_containing_module(node.node_name)

            if hasattr(nncf_module, "pre_ops"):
                for key in list(nncf_module.pre_ops.keys()):
                    op = nncf_module.get_pre_op(key)
                    if isinstance(op.op, FilterPruningMask):
                        nncf_module.remove_pre_forward_operation(key)

            if hasattr(nncf_module, "post_ops"):
                for key in list(nncf_module.post_ops.keys()):
                    op = nncf_module.post_ops(key)
                    if isinstance(op.op, FilterPruningMask):
                        nncf_module.remove_post_forward_operation(key)

    def prune_model(self):
        """
        Model pruner work in two stages:
        1. Mask propagation: propagate pruning masks through the graph.
        2. Applying calculated masks.
        3. Remove filter pruning operations.
        """
        nncf_logger.info("Start pruning model")
        self.mask_propagation()
        self.apply_mask()
        self.remove_filter_pruning_operations()
        nncf_logger.info("Finished pruning model")
