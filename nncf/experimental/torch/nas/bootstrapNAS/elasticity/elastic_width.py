"""
 Copyright (c) 2022 Intel Corporation
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
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from nncf.common.graph import BaseLayerAttributes
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.node_selector import PruningNodeSelector
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.pruning.utils import get_cluster_next_nodes
from nncf.common.pruning.utils import get_conv_in_out_channels
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.tensor import NNCFTensor
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.operator_metatypes import PTBatchNormMetatype
from nncf.torch.graph.operator_metatypes import PTConv1dMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTConv3dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv1dSubtype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv3dSubtype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdateBatchNormParams
from nncf.torch.module_operations import UpdateNumGroups
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.module_operations import UpdateWeightAndOptionalBias
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_HANDLERS_MAP
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.filter_reorder import FilterReorderingAlgorithm
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from nncf.torch.pruning.operations import PTElementwisePruningOp
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.tensor_processor import PTNNCFPruningTensorProcessor
from nncf.torch.pruning.utils import collect_input_shapes
from nncf.torch.pruning.utils import collect_output_shapes
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.utils import get_filters_num

pruning_group_id = int
width_type = int
ElasticWidthConfig = Dict[pruning_group_id, width_type]
WidthList = List[width_type]
ElasticWidthSearchSpace = Dict[pruning_group_id, WidthList]


class DynamicWidthOp:
    """
    Base class for operations that dynamically adapt the layer parameters after removing some
    number of filters of preceding layers (input width) or the current layer (output width).
    """

    def __init__(self, *args, max_width: int, node_name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_width = max_width
        self._max_width = max_width
        self._node_name = node_name

    @property
    def max_width(self) -> width_type:
        return self._max_width

    def get_active_width(self) -> width_type:
        return self._active_width

    def set_active_width(self, width: width_type):
        """
        Set new number of filters that was set for preceding layers (input width) or for current layer
        (output width). The operation will use this value to adapt parameters for the layer, to which the operation is
        applied. The width should be less the original number of filters and more than one. Zero number of filters is
        supported through Dynamic Depth feature.

        :param width: number of filters that was set for preceding layers.
        """
        if width is None or width > self._max_width or width < 1:
            raise AttributeError('Invalid width={} in scope={}.\nIt should be within the range: [1, {}]'.format(
                width, self._node_name, self._max_width))

        self._active_width = width


class ElasticWidthOp(DynamicWidthOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._width_list = []

    @property
    def width_list(self) -> List[int]:
        return self._width_list


class ElasticWidthInfo(PrunedLayerInfoBase):
    def __init__(self,
                 node_name: NNCFNodeName,
                 module: nn.Module,
                 elastic_op: ElasticWidthOp,
                 node_id: int,
                 is_depthwise: bool):
        super().__init__(node_name, node_id, is_depthwise)
        self.module = module
        self.elastic_op = elastic_op


class DynamicLinearInputOp(DynamicWidthOp, nn.Module):
    def forward(self, weight):
        return weight[:, :self._active_width]


class ElasticLinearOp(DynamicWidthOp, nn.Module):
    def forward(self, weight, bias):
        new_bias = None if bias is None else bias[:self._active_width].contiguous()
        return weight[:self._active_width, :].contiguous(), new_bias


class DynamicConvInputOp(DynamicWidthOp, nn.Module):
    def forward(self, weight):
        return weight[:, :self._active_width, :, :]


class DynamicDWConvInputOp(DynamicWidthOp, nn.Module):
    def forward(self, _):
        return self._active_width


class DynamicBatchNormInputOp(DynamicWidthOp, nn.Module):
    SET_RUNNING_STATISTICS = False

    def forward(self, **bn_params):
        return [param[:self._active_width] for param in bn_params.values()]


class ElasticWidthConv2DOp(ElasticWidthOp, nn.Module):
    def __init__(self, max_out_channels, node_name, elastic_width_params: Dict[str, Any] = None):
        super().__init__(max_width=max_out_channels, node_name=node_name)

        if elastic_width_params is None:
            elastic_width_params = {}
        min_out_channels = elastic_width_params.get('min_out_channels', 32)
        max_num_params = elastic_width_params.get('max_num_widths', -1)
        width_step = elastic_width_params.get('width_step', 32)
        width_multipliers = elastic_width_params.get('width_multipliers', [])

        self._width_list = self.generate_width_list(self._max_width, min_out_channels, max_num_params, width_step,
                                                    width_multipliers)

    def set_active_width(self, width: int):
        if width not in self.width_list and width != self.max_width:
            raise ValueError(f'Invalid number of output channels to set: {width} in scope={self._node_name}. '
                             f'Should be a number in {self.width_list}')
        super().set_active_width(width)

    def forward(self, weight, bias):
        nncf_logger.debug('Conv2d with active width={} in scope={}'.format(self._active_width, self._node_name))
        num_out_channels = self._active_width
        new_bias = None if bias is None else bias[:num_out_channels]
        new_weights = weight[:num_out_channels, :, :, :]
        return [new_weights, new_bias]

    @staticmethod
    def generate_width_list(max_out_channels, min_out_channels, max_num_params, width_step, width_multipliers):
        width_list = []
        if max_out_channels <= min_out_channels:
            width_list.append(max_out_channels)
        elif not width_multipliers:
            width = max_out_channels
            width_list.append(width)
            width -= width_step
            while width >= min_out_channels:
                if max_num_params == len(width_list):
                    break
                width_list.append(width)
                width -= width_step
        else:
            width_multipliers.sort(reverse=True)
            if width_multipliers[0] < 1:
                width_list.append(max_out_channels)
            for multiplier in width_multipliers:
                if max_num_params == len(width_list):
                    break
                if 0 >= multiplier > 1:
                    nncf_logger.warning("Wrong value for multiplier: {}. Skipping ".format(multiplier))
                    continue
                w = int(max_out_channels * multiplier)
                w = w - (w % 8)
                w = max(w, min_out_channels)
                if w in width_list:
                    continue
                width_list.append(w)
        return width_list


class ElasticWidthHandler(SingleElasticityHandler):
    """
    An interface for handling elastic width dimension in the network, i.e. define number of channels in the layers.
    """

    def __init__(self, target_model: NNCFNetwork,
                 filter_importance: Callable,
                 weights_normalizer: Optional[Callable],
                 node_name_vs_dynamic_input_width_op_map: Dict[NNCFNodeName, DynamicWidthOp],
                 pruned_module_groups_info: Clusterization[ElasticWidthInfo](id_fn=lambda x: x.node_name),
                 transformation_commands: List[TransformationCommand]):
        super().__init__()
        self._target_model = target_model
        self._node_name_vs_dynamic_input_width_op_map = node_name_vs_dynamic_input_width_op_map
        self._pruned_module_groups_info = pruned_module_groups_info
        self._transformation_commands = transformation_commands
        self._filter_importance = filter_importance
        self._weights_normalizer = weights_normalizer

        graph = self._target_model.get_original_graph()
        prunable_types = [NNCFConv2d.op_func_name]
        self._cluster_next_nodes = get_cluster_next_nodes(graph, self._pruned_module_groups_info, prunable_types)

        self._propagation_graph = deepcopy(graph)

        self._width_num_params_indicator = -1

    @property
    def width_num_params_indicator(self):
        return self._width_num_params_indicator

    @width_num_params_indicator.setter
    def width_num_params_indicator(self, width_num_params_indicator):
        if width_num_params_indicator == 0 or width_num_params_indicator < -1:
            raise RuntimeError(f"Invalid width indicator: {width_num_params_indicator}")
        self._width_num_params_indicator = width_num_params_indicator

    @property
    def propagation_graph(self):
        return self._propagation_graph

    def get_transformation_commands(self) -> List[TransformationCommand]:
        """
        :return: transformation commands for introducing the elasticity to NNCFNetwork
        """
        return self._transformation_commands

    def get_search_space(self) -> ElasticWidthSearchSpace:
        if self._width_num_params_indicator == -1:
            return self._collect_ops_data_by_selection_rule(lambda op: op.width_list)
        return self._collect_ops_data_by_selection_rule(
            lambda op: op.width_list[:min(self._width_num_params_indicator, len(op.width_list))])

    def get_active_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes currently activated Subnet

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(lambda op: op.get_active_width())

    def get_random_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen width for each elastic layer

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(
            lambda op: op.width_list[random.randrange(0, self._get_width_list_len(op))]
        )

    def get_minimum_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum width in each elastic layer

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(lambda op: min(self._get_width_list(op)))

    def get_maximum_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum width in each elastic layer

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(lambda op: max(self._get_width_list(op)))

    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network to which elasticity was applied.
        """
        supernet_config = self._collect_ops_data_by_selection_rule(lambda op: op.max_width)
        self.set_config(supernet_config)

    def set_config(self, config: ElasticWidthConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :param config: map of pruning group id to width value
        """
        for node in self._propagation_graph.get_all_nodes():
            node.data.pop('output_mask', None)

        names_of_processed_nodes = set()
        for cluster_id, width in config.items():
            cluster = self._pruned_module_groups_info.get_cluster_by_id(cluster_id)
            for elastic_width_info in cluster.elements:
                node_id = elastic_width_info.nncf_node_id
                node = self._propagation_graph.get_node_by_id(node_id)
                max_width = elastic_width_info.elastic_op.max_width
                mask = self._width_to_mask(max_width, width)
                node.data['output_mask'] = mask
                elastic_width_info.elastic_op.set_active_width(width)
                names_of_processed_nodes.add(node_id)

        algo = MaskPropagationAlgorithm(
            self._propagation_graph, PT_PRUNING_OPERATOR_METATYPES, PTNNCFPruningTensorProcessor)
        algo.mask_propagation()

        for node_name, dynamic_input_width_op in self._node_name_vs_dynamic_input_width_op_map.items():
            node = self._propagation_graph.get_node_by_name(node_name)
            input_masks = get_input_masks(node, self._propagation_graph)
            was_set = False
            if input_masks:
                input_mask = input_masks[0]
                input_width = self.mask_to_width(input_mask)
                if input_width:
                    dynamic_input_width_op.set_active_width(input_width)
                    was_set = True

            if not was_set and node_name not in names_of_processed_nodes:
                nncf_logger.debug('input width was not set in scope={}'.format(node.node_name))

    def get_kwargs_for_flops_counting(self) -> Dict[str, Any]:
        graph = self._target_model.get_graph()
        modules_out_shapes = collect_output_shapes(graph)
        modules_in_shapes = collect_input_shapes(graph)

        GENERAL_CONV_LAYER_METATYPES = [
            PTConv1dMetatype,
            PTDepthwiseConv1dSubtype,
            PTConv2dMetatype,
            PTDepthwiseConv2dSubtype,
            PTConv3dMetatype,
            PTDepthwiseConv3dSubtype,
            PTConvTranspose2dMetatype,
            PTConvTranspose3dMetatype
        ]
        LINEAR_LAYER_METATYPES = [
            PTLinearMetatype
        ]

        tmp_in_channels, tmp_out_channels = get_conv_in_out_channels(graph)

        for group in self._pruned_module_groups_info.get_all_clusters():
            assert all(tmp_out_channels[group.elements[0].node_name] == tmp_out_channels[node.node_name]
                       for node in group.elements)
            first_elastic_width_info = group.elements[0]  # type: ElasticWidthInfo
            first_elastic_op = first_elastic_width_info.elastic_op  # type: ElasticWidthConv2DOp
            new_out_channels_num = first_elastic_op.get_active_width()
            num_of_pruned_elems = first_elastic_op.max_width - new_out_channels_num
            for elastic_width_info in group.elements:
                tmp_out_channels[elastic_width_info.node_name] = new_out_channels_num

            # Prune in_channels in all next nodes of cluster
            for node_name in self._cluster_next_nodes[group.id]:
                tmp_in_channels[node_name] -= num_of_pruned_elems

        return {
            'graph': graph,
            'input_shapes': modules_in_shapes,
            'output_shapes': modules_out_shapes,
            'input_channels': tmp_in_channels,
            'output_channels': tmp_out_channels,
            'conv_op_metatypes': GENERAL_CONV_LAYER_METATYPES,
            'linear_op_metatypes': LINEAR_LAYER_METATYPES,
        }

    def resolve_conflicts_with_other_elasticities(self,
                                                  config: ElasticWidthConfig,
                                                  elasticity_handlers: ELASTICITY_HANDLERS_MAP) -> ElasticWidthConfig:
        """
        Resolves a conflict between the given elasticity config and active elasticity configs of the given handlers.
        For example, elastic width configuration may contradict to elastic depth one. When we activate some
        configuration in the Elastic Width Handler, i.e. define number of output channels for some layers, we
        change output shapes of the layers. Consequently, it affects the blocks that can be skipped by Elastic Depth
        Handler, because input and output shapes may not be identical now.

        :param config: elasticity configuration
        :param elasticity_handlers: map of elasticity dimension to elasticity handler
        :return: elasticity configuration without conflicts with other active configs of other elasticity handlers
        """
        return config

    def get_group_id_by_node_name(self, node_name: NNCFNodeName) -> Optional[int]:
        group_id = None
        for cluster in self._pruned_module_groups_info.get_all_clusters():
            for element in cluster.elements:
                if node_name == element.node_name:
                    group_id = cluster.id
        return group_id

    def reorganize_weights(self):
        for node in self._propagation_graph.get_all_nodes():
            node.data.pop('output_mask', None)

        # 1. Calculate filter importance for all groups of prunable layers
        for group in self._pruned_module_groups_info.get_all_clusters():
            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in group.elements])
            assert torch.all(filters_num == filters_num[0])
            device = group.elements[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # 1.1 Calculate cumulative importance for all filters in this group
            for minfo in group.elements:
                weight = minfo.module.weight
                if self._weights_normalizer:
                    weight = self._weights_normalizer(minfo.module.weight)
                filters_importance = self._filter_importance(weight,
                                                             minfo.module.target_weight_dim_for_compression)
                cumulative_filters_importance += filters_importance

            _, reorder_indexes = torch.sort(cumulative_filters_importance, dim=0, descending=True)

            # 1.2 Setup reorder indexes as output mask to reorganize filters
            for minfo in group.elements:
                node = self._propagation_graph.get_node_by_id(minfo.nncf_node_id)
                node.data['output_mask'] = PTNNCFTensor(reorder_indexes)

        # 2. Propagating masks across the graph
        reorder_algo = FilterReorderingAlgorithm(self._target_model,
                                                 self._propagation_graph,
                                                 PT_PRUNING_OPERATOR_METATYPES,
                                                 PTNNCFPruningTensorProcessor)
        reorder_algo.reorder_filters()

    def find_pairs_of_nodes_with_different_width(self, pairs_of_nodes: List[Tuple[str, str]]) -> List[int]:
        pair_indexes = []
        for idx, (start_node_name, end_node_name) in enumerate(pairs_of_nodes):
            start_node = self._propagation_graph.get_node_by_name(start_node_name)
            start_mask = start_node.data['output_mask']
            start_width = ElasticWidthHandler.mask_to_width(start_mask)
            end_node = self._propagation_graph.get_node_by_name(end_node_name)
            end_mask = end_node.data['output_mask']
            end_width = ElasticWidthHandler.mask_to_width(end_mask)
            output_node_names = [node.node_name for node in self._propagation_graph.get_output_nodes()]
            if start_width != end_width and end_node_name not in output_node_names:
                pair_indexes.append(idx)
                nncf_logger.warning('Pair of nodes [{}, {}] has a different width: {} != {}'.format(
                    start_node_name, end_node_name, start_width, end_width))
        return pair_indexes

    def _get_width_list_len(self, op: ElasticWidthOp) -> int:
        N = len(op.width_list)
        if 0 < self._width_num_params_indicator < N:
            return self._width_num_params_indicator
        return N

    def _get_width_list(self, op: ElasticWidthOp) -> List[int]:
        width_list_len = self._get_width_list_len(op)
        return op.width_list[:width_list_len]

    def _collect_ops_data_by_selection_rule(self, selection_rule: Callable) -> Dict[pruning_group_id, Any]:
        elastic_width_config = {}
        for cluster in self._pruned_module_groups_info.get_all_clusters():
            all_max_out_channels = {el.elastic_op.max_width for el in cluster.elements}
            if len(all_max_out_channels) != 1:
                raise RuntimeError('Invalid grouping of layers with different number of output channels')

            first_elastic_width_info = next(iter(cluster.elements))
            op = first_elastic_width_info.elastic_op
            selected_width = selection_rule(op)
            elastic_width_config[cluster.id] = selected_width
            nncf_logger.debug('Select width={} for group #{}'.format(cluster.id, selected_width))
        return elastic_width_config

    @staticmethod
    def mask_to_width(mask: NNCFTensor) -> Optional[int]:
        result = None
        if mask is not None:
            actual_mask = mask.tensor
            mask_len = sum(actual_mask.size())
            width = int(sum(actual_mask))
            ref_mask = ElasticWidthHandler._width_to_mask(mask_len, width).tensor
            assert torch.equal(ref_mask, actual_mask), \
                f'Invalid mask {actual_mask}: the first {width} values must be ones, the rest - zeros.'
            result = width
        return result

    @staticmethod
    def _width_to_mask(active_width: int, max_width: int) -> PTNNCFTensor:
        mask = torch.ones(max_width)
        mask[active_width:].fill_(0)
        return PTNNCFTensor(mask)


class EWBuilderStateNames:
    GROUPED_NODE_NAMES_TO_PRUNE = 'grouped_node_names_to_prune'


@ELASTICITY_BUILDERS.register(ElasticityDim.WIDTH)
class ElasticWidthBuilder(SingleElasticityBuilder):
    _state_names = EWBuilderStateNames

    def __init__(self, elasticity_params: Optional[Dict[str, Any]] = None,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None):
        super().__init__(ignored_scopes, target_scopes, elasticity_params)
        self._weights_normalizer = None
        self._grouped_node_names_to_prune = []  # type: List[List[NNCFNodeName]]

    def build(self, target_model: NNCFNetwork) -> ElasticWidthHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elastic width and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elastic width.
        """
        filter_importance_str = self._elasticity_params.get('filter_importance', 'L1')
        filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(filter_importance_str)

        graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device

        if not self._grouped_node_names_to_prune:
            prunable_types = [NNCFConv2d.op_func_name]
            types_of_grouping_ops = PTElementwisePruningOp.get_all_op_aliases()
            pruning_node_selector = PruningNodeSelector(PT_PRUNING_OPERATOR_METATYPES,
                                                        prunable_types,
                                                        types_of_grouping_ops,
                                                        ignored_scopes=self._ignored_scopes,
                                                        target_scopes=self._target_scopes,
                                                        prune_first=True,
                                                        prune_downsample_convs=True)
            groups_of_nodes_to_prune = pruning_node_selector.create_pruning_groups(graph)
            for group in groups_of_nodes_to_prune.get_all_clusters():
                grouped_node_names = [node.node_name for node in group.elements]
                self._grouped_node_names_to_prune.append(grouped_node_names)

        transformation_commands = []
        pruned_module_groups_info = Clusterization[ElasticWidthInfo](id_fn=lambda x: x.node_name)
        node_name_vs_dynamic_input_width_op_map = OrderedDict()

        for i, grouped_node_names in enumerate(self._grouped_node_names_to_prune):
            group_minfos = []
            list_of_node_ids = []
            for node_name in grouped_node_names:
                node = graph.get_node_by_name(node_name)
                list_of_node_ids.append(node.node_id)
                layer_attrs = node.layer_attributes
                elastic_width_operation = self._create_elastic_width_op(layer_attrs, node_name, self._elasticity_params)
                elastic_width_operation.to(device)
                update_conv_params_op = UpdateWeightAndOptionalBias(elastic_width_operation)
                transformation_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(
                            TargetType.PRE_LAYER_OPERATION,
                            target_node_name=node_name
                        ),
                        update_conv_params_op,
                        TransformationPriority.PRUNING_PRIORITY
                    )
                )
                conv_module = target_model.get_containing_module(node_name)
                assert isinstance(conv_module, nn.Conv2d), 'currently prune only 2D Convolutions'

                group_minfos.append(ElasticWidthInfo(node_name=node_name,
                                                     module=conv_module,
                                                     elastic_op=elastic_width_operation,
                                                     node_id=node.node_id,
                                                     is_depthwise=is_prunable_depthwise_conv(node)))

            cluster = Cluster[ElasticWidthInfo](i, group_minfos, list_of_node_ids)
            pruned_module_groups_info.add_cluster(cluster)

        metatype_vs_op_creator = {
            PTConv2dMetatype: self._create_dynamic_conv_input_op,
            PTDepthwiseConv2dSubtype: self._create_dynamic_dw_conv_input_op,
            PTBatchNormMetatype: self._create_dynamic_bn_input_op,
            PTLinearMetatype: self._create_dynamic_linear_input_op
        }
        for metatype, op_creator in metatype_vs_op_creator.items():
            nodes = graph.get_nodes_by_metatypes([metatype])
            for node in nodes:
                node_name = node.node_name
                nncf_logger.info("Adding Dynamic Input Op for {} in scope: {}".format(metatype.name, node_name))
                layer_attrs = node.layer_attributes
                update_module_params = op_creator(layer_attrs, node_name).to(device)
                node_name_vs_dynamic_input_width_op_map[node_name] = update_module_params.op
                transformation_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(
                            TargetType.PRE_LAYER_OPERATION,
                            target_node_name=node_name
                        ),
                        update_module_params,
                        priority=TransformationPriority.DEFAULT_PRIORITY
                    )
                )

        return ElasticWidthHandler(target_model, filter_importance, self._weights_normalizer,
                                   node_name_vs_dynamic_input_width_op_map,
                                   pruned_module_groups_info, transformation_commands)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        super().load_state(state)
        self._grouped_node_names_to_prune = state[self._state_names.GROUPED_NODE_NAMES_TO_PRUNE]

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        state = super().get_state()
        state[self._state_names.GROUPED_NODE_NAMES_TO_PRUNE] = self._grouped_node_names_to_prune
        return state

    @staticmethod
    def _create_elastic_width_op(conv_layer_attrs: BaseLayerAttributes, node_name: str, elastic_width_params):
        assert isinstance(conv_layer_attrs, ConvolutionLayerAttributes)
        nncf_logger.info("Adding Dynamic Conv2D Layer in scope: {}".format(str(node_name)))
        return ElasticWidthConv2DOp(conv_layer_attrs.out_channels, node_name, elastic_width_params)

    @staticmethod
    def _create_dynamic_conv_input_op(conv_layer_attrs: BaseLayerAttributes, node_name: str):
        assert isinstance(conv_layer_attrs, ConvolutionLayerAttributes)
        dynamic_conv_input_op = DynamicConvInputOp(max_width=conv_layer_attrs.in_channels, node_name=node_name)
        return UpdateWeight(dynamic_conv_input_op)

    @staticmethod
    def _create_dynamic_dw_conv_input_op(conv_layer_attrs: BaseLayerAttributes, node_name: str):
        assert isinstance(conv_layer_attrs, ConvolutionLayerAttributes)
        dynamic_dw_conv_input_op = DynamicDWConvInputOp(max_width=conv_layer_attrs.groups, node_name=node_name)
        return UpdateNumGroups(dynamic_dw_conv_input_op)

    @staticmethod
    def _create_dynamic_bn_input_op(generic_layer_attrs: BaseLayerAttributes, node_name: str):
        assert isinstance(generic_layer_attrs, GenericWeightedLayerAttributes)
        dynamic_bn_input_op = DynamicBatchNormInputOp(max_width=generic_layer_attrs.get_num_filters(),
                                                      node_name=node_name)
        return UpdateBatchNormParams(dynamic_bn_input_op)

    @staticmethod
    def _create_dynamic_linear_input_op(linear_layer_attrs: BaseLayerAttributes, node_name: str):
        assert isinstance(linear_layer_attrs, LinearLayerAttributes)
        dynamic_linear_input_op = DynamicLinearInputOp(max_width=linear_layer_attrs.in_features, node_name=node_name)
        return UpdateWeight(dynamic_linear_input_op)
