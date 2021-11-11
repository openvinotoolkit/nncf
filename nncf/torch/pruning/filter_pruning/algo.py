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

import json
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.common.pruning.statistics import FilterPruningStatistics
from nncf.common.pruning.statistics import PrunedModelTheoreticalBorderline
from nncf.common.pruning.statistics import PrunedLayerSummary
from nncf.common.pruning.statistics import PrunedModelStatistics
from nncf.common.pruning.utils import calculate_in_out_channels_in_uniformly_pruned_model
from nncf.common.pruning.utils import count_flops_and_weights
from nncf.common.pruning.utils import count_flops_and_weights_per_node
from nncf.common.pruning.utils import get_cluster_next_nodes
from nncf.common.pruning.utils import get_conv_in_out_channels
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.debug import is_debug
from nncf.common.utils.logger import logger as nncf_logger
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.operator_metatypes import Conv1dMetatype
from nncf.torch.graph.operator_metatypes import Conv2dMetatype
from nncf.torch.graph.operator_metatypes import Conv3dMetatype
from nncf.torch.graph.operator_metatypes import ConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import ConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import DepthwiseConv1dSubtype
from nncf.torch.graph.operator_metatypes import DepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import DepthwiseConv3dSubtype
from nncf.torch.graph.operator_metatypes import LinearMetatype
from nncf.torch.layers import NNCF_GENERAL_CONV_MODULES_DICT
from nncf.torch.layers import NNCF_LINEAR_MODULES_DICT
from nncf.torch.layers import NNCF_PRUNING_MODULES_DICT
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.base_algo import BasePruningAlgoBuilder
from nncf.torch.pruning.base_algo import BasePruningAlgoController
from nncf.torch.pruning.operations import PTElementwisePruningOp
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from nncf.torch.pruning.filter_pruning.functions import calculate_binary_mask
from nncf.torch.pruning.filter_pruning.functions import tensor_l2_normalizer
from nncf.torch.pruning.filter_pruning.global_ranking.legr import LeGR
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.pruning.structs import PrunedModuleInfo
from nncf.torch.pruning.utils import init_output_masks_in_graph
from nncf.torch.structures import LeGRInitArgs, DistributedCallbacksArgs
from nncf.torch.utils import get_filters_num

GENERAL_CONV_LAYER_METATYPES = [
    Conv1dMetatype,
    DepthwiseConv1dSubtype,
    Conv2dMetatype,
    DepthwiseConv2dSubtype,
    Conv3dMetatype,
    DepthwiseConv3dSubtype,
    ConvTranspose2dMetatype,
    ConvTranspose3dMetatype
]
LINEAR_LAYER_METATYPES = [
    LinearMetatype
]


@PT_COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    def create_weight_pruning_operation(self, module, node_name):
        return FilterPruningMask(module.weight.size(module.target_weight_dim_for_compression),
                                 node_name, module.target_weight_dim_for_compression)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return FilterPruningController(model,
                                       self._prunable_types,
                                       self.pruned_module_groups_info,
                                       self._pruned_norms_operators,
                                       self.config)

    def _is_pruned_module(self, module) -> bool:
        # Currently prune only Convolutions
        return isinstance(module, tuple(NNCF_PRUNING_MODULES_DICT.keys()))

    def get_op_types_of_pruned_modules(self) -> List[str]:
        types = [v.op_func_name for v in NNCF_PRUNING_MODULES_DICT]
        return types

    def get_types_of_grouping_ops(self) -> List[str]:
        return PTElementwisePruningOp.get_all_op_aliases()


@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_filter_pruning')
class FilterPruningController(BasePruningAlgoController):
    def __init__(self, target_model: NNCFNetwork,
                 prunable_types: List[str],
                 pruned_module_groups: Clusterization[PrunedModuleInfo],
                 pruned_norms_operators: List[Tuple[NNCFNode, FilterPruningMask, torch.nn.Module]],
                 config: NNCFConfig):
        #pylint:disable=too-many-statements
        super().__init__(target_model, prunable_types, pruned_module_groups, config)
        params = self.pruning_config.get('params', {})
        self._pruned_norms_operators = pruned_norms_operators
        self.frozen = False
        self._pruning_rate = 0
        self.pruning_init = self.pruning_config.get('pruning_init', 0)
        self.pruning_quota = 0.9
        self.normalize_weights = True

        self._init_module_channels_and_shapes()
        self.pruning_quotas = {}
        self.nodes_flops = {}  # type: Dict[NNCFNodeName, int]
        self.nodes_params_num = {}  # type: Dict[NNCFNodeName, int]
        self.next_nodes = {}  # type: Dict[int, List[NNCFNodeName]]
        self._init_pruned_modules_params()
        self.flops_count_init()
        self.full_flops = sum(self.nodes_flops.values())
        self.current_flops = self.full_flops
        self.full_params_num = sum(self.nodes_params_num.values())
        self.current_params_num = self.full_params_num
        self._pruned_layers_num = len(self.pruned_module_groups_info.get_all_nodes())
        self._prunable_layers_num = len(self._model.get_graph().get_nodes_by_types(self._prunable_types))
        self._max_prunable_flops, self._max_prunable_params =\
            self._calculate_flops_and_weights_in_uniformly_pruned_model(1.)

        self.weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self.filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(params.get('filter_importance', 'L2'))
        self.ranking_type = params.get('interlayer_ranking_type', 'unweighted_ranking')
        self.all_weights = params.get("all_weights", False)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get('schedule', 'exponential'))
        self._scheduler = scheduler_cls(self, params)

        if self.ranking_type == 'learned_ranking':
            # In case of learned_ranking ranking type weights shouldn't be normalized
            self.normalize_weights = False
            if params.get('load_ranking_coeffs_path'):
                coeffs_path = params.get('load_ranking_coeffs_path')
                nncf_logger.info('Loading ranking coefficients from file {}'.format(coeffs_path))
                try:
                    with open(coeffs_path, 'r', encoding='utf8') as coeffs_file:
                        loaded_coeffs = json.load(coeffs_file)
                except (ValueError, FileNotFoundError) as err:
                    raise Exception('Can\'t load json with ranking coefficients. Please, check format of json file '
                                    'and path to the file.') from err
                ranking_coeffs = {key: tuple(loaded_coeffs[key]) for key in loaded_coeffs}
                nncf_logger.info('Loaded ranking coefficients = {}'.format(ranking_coeffs))
                self.ranking_coeffs = ranking_coeffs
            else:
                # Ranking can't be trained without registered init struct LeGRInitArgs
                if not config.has_extra_struct(LeGRInitArgs):
                    raise Exception('Please, register LeGRInitArgs via register_default_init_args function.')
                # Wrapping model for parallelization
                distributed_wrapping_init_args = config.get_extra_struct(DistributedCallbacksArgs)
                target_model = distributed_wrapping_init_args.wrap_model(target_model)
                legr_init_args = config.get_extra_struct(LeGRInitArgs)
                legr_params = params.get("legr_params", {})
                if 'max_pruning' not in legr_params:
                    legr_params['max_pruning'] = self._scheduler.target_level
                self.legr = LeGR(self, target_model, legr_init_args, **legr_params)
                self.ranking_coeffs = self.legr.train_global_ranking()
                nncf_logger.info('Trained ranking coefficients = {}'.format(self.ranking_coeffs))
                # Unwrapping parallelized model
                target_model = distributed_wrapping_init_args.unwrap_model(target_model)
        else:
            self.ranking_coeffs = {node.node_name: (1, 0) for node in self.pruned_module_groups_info.get_all_nodes()}

        # Saving ranking coefficients to the specified file
        if params.get('save_ranking_coeffs_path'):
            nncf_logger.info(
                'Saving ranking coefficients to the file {}'.format(params.get('save_ranking_coeffs_path')))
            with open(params.get('save_ranking_coeffs_path'), 'w', encoding='utf8') as f:
                json.dump(self.ranking_coeffs, f)

        self.set_pruning_rate(self.pruning_init)
        self._bn_adaptation = None

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @staticmethod
    def get_mask(minfo: PrunedModuleInfo) -> torch.Tensor:
        return minfo.operand.binary_filter_pruning_mask

    @staticmethod
    def set_mask(minfo: PrunedModuleInfo, mask: torch.Tensor) -> None:
        minfo.operand.binary_filter_pruning_mask = mask

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        if not quickly_collected_only and is_debug():
            stats = PrunedModelTheoreticalBorderline(
                self._pruned_layers_num, self._prunable_layers_num, self._max_prunable_flops,
                self._max_prunable_params, self.full_flops, self.full_params_num)

            nncf_logger.debug(stats.to_str())

        pruned_layers_summary = {}
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            layer_name = str(minfo.module_scope)
            if layer_name not in pruned_layers_summary:
                pruned_layers_summary[layer_name] = \
                    PrunedLayerSummary(layer_name,
                                       list(minfo.module.weight.size()),
                                       list(self.mask_shape(minfo)),
                                       self.pruning_rate_for_mask(minfo))

        model_statistics = PrunedModelStatistics(self._pruning_rate, list(pruned_layers_summary.values()))
        self._update_benchmark_statistics()
        target_pruning_level = self.scheduler.current_pruning_level

        stats = FilterPruningStatistics(model_statistics, self.full_flops, self.current_flops,
                                        self.full_params_num, self.current_params_num, target_pruning_level)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('filter_pruning', stats)
        return nncf_stats

    @property
    def pruning_rate(self) -> float:
        """Global pruning rate in the model"""
        return self._pruning_rate

    def freeze(self, freeze: bool = True):
        self.frozen = freeze

    def _init_module_channels_and_shapes(self):
        self._modules_in_channels = {}  # type: Dict[NNCFNodeName, int]
        self._modules_out_channels = {}  # type: Dict[NNCFNodeName, int]
        self._modules_in_shapes = {}  # type: Dict[NNCFNodeName, List[int]]
        self._modules_out_shapes = {}  # type: Dict[NNCFNodeName, List[int]]

    def _init_pruned_modules_params(self):
        # 1. Init in/out channels for potentially prunable modules
        graph = self._model.get_original_graph()
        self._modules_in_channels, self._modules_out_channels = get_conv_in_out_channels(graph)

        # 2. Init next_nodes for every pruning cluster
        self.next_nodes = get_cluster_next_nodes(graph, self.pruned_module_groups_info, self._prunable_types)

        # 3. Init pruning quotas
        for cluster in self.pruned_module_groups_info.get_all_clusters():
            self.pruning_quotas[cluster.id] = np.floor(self._modules_out_channels[cluster.elements[0].node_name] \
                                                       * self.pruning_quota)

    def _calculate_output_shape(self, graph: NNCFGraph, node: NNCFNode) -> Tuple[int, ...]:
        """
        Calculates output shape of convolution layer by input edge.

        :param graph: the model graph
        :param node: node from NNCF graph
        :return: output shape
        """
        in_edge = graph.get_input_edges(node)[0]
        shape = list(in_edge.tensor_shape)[2:]
        attrs = node.layer_attributes

        assert isinstance(attrs, ConvolutionLayerAttributes)

        for i, _ in enumerate(shape):
            if attrs.transpose:
                shape[i] = (shape[i] - 1) * attrs.stride[i] - 2 * attrs.padding_values[i] + attrs.kernel_size[i]
            else:
                shape[i] = (shape[i] + 2 * attrs.padding_values[i] - attrs.kernel_size[i]) // attrs.stride[i] + 1
        return tuple(shape)

    def flops_count_init(self) -> None:
        graph = self._model.get_original_graph()
        for node in graph.get_nodes_by_types([v.op_func_name for v in NNCF_GENERAL_CONV_MODULES_DICT]):
            output_edges = graph.get_output_edges(node)
            if output_edges:
                out_edge = output_edges[0]
                out_shape = out_edge.tensor_shape[2:]
            else:
                # For disconnected NNCFGraph when node have no output edge
                out_shape = self._calculate_output_shape(graph, node)
                nncf_logger.error("Node %s have no output edge in NNCFGraph", node.node_name)
            self._modules_out_shapes[node.node_name] = out_shape

        for node in graph.get_nodes_by_types([v.op_func_name for v in NNCF_LINEAR_MODULES_DICT]):
            output_edges = graph.get_output_edges(node)
            if output_edges:
                out_edge = graph.get_output_edges(node)[0]
                out_shape = out_edge.tensor_shape
                self._modules_out_shapes[node.node_name] = out_shape[-1]
            else:
                # For disconnected NNCFGraph when node have no output edge
                nncf_logger.error("Node %s have no output edge in NNCFGraph", node.node_name)
                self._modules_out_shapes[node.node_name] = node.layer_attributes.out_features

            in_edge = graph.get_input_edges(node)[0]
            in_shape = in_edge.tensor_shape
            if len(in_shape) == 1:
                self._modules_in_shapes[node.node_name] = in_shape[0]
            else:
                self._modules_in_shapes[node.node_name] = in_shape[1:]

        self.nodes_flops, self.nodes_params_num = \
            count_flops_and_weights_per_node(graph, self._modules_in_shapes, self._modules_out_shapes,
                                             conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                             linear_op_metatypes=LINEAR_LAYER_METATYPES)

    def _calculate_flops_and_weights_pruned_model_by_masks(self) -> Tuple[int, int]:
        """
        Calculates number of weights and flops for pruned model by using binary_filter_pruning_mask.

        :return: number of flops in model
        """
        tmp_in_channels = self._modules_in_channels.copy()
        tmp_out_channels = self._modules_out_channels.copy()

        for group in self.pruned_module_groups_info.get_all_clusters():
            assert all(tmp_out_channels[group.elements[0].node_name] == tmp_out_channels[node.node_name]
                       for node in group.elements)
            new_out_channels_num = int(sum(group.elements[0].operand.binary_filter_pruning_mask))
            num_of_sparse_elems = len(group.elements[0].operand.binary_filter_pruning_mask) - new_out_channels_num
            for node in group.elements:
                tmp_out_channels[node.node_name] = new_out_channels_num
            # Prune in_channels in all next nodes of cluster
            next_nodes = self.next_nodes[group.id]
            for node_name in next_nodes:
                tmp_in_channels[node_name] -= num_of_sparse_elems

        return count_flops_and_weights(self._model.get_original_graph(),
                                       self._modules_in_shapes,
                                       self._modules_out_shapes,
                                       input_channels=tmp_in_channels,
                                       output_channels=tmp_out_channels,
                                       conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                       linear_op_metatypes=LINEAR_LAYER_METATYPES)

    def _calculate_flops_and_weights_in_uniformly_pruned_model(self, pruning_rate: float) -> Tuple[int, int]:
        """
        Prune all prunable modules in model with pruning_rate rate and returns number of weights and
        flops of the pruned model.

        :param pruning_rate: proportion of zero filters in all modules
        :return: flops number in pruned model
        """
        tmp_in_channels, tmp_out_channels = \
            calculate_in_out_channels_in_uniformly_pruned_model(
                pruning_groups=self.pruned_module_groups_info.get_all_clusters(),
                pruning_rate=pruning_rate,
                full_input_channels=self._modules_in_channels,
                full_output_channels=self._modules_out_channels,
                pruning_groups_next_nodes=self.next_nodes)

        return count_flops_and_weights(self._model.get_original_graph(),
                                       self._modules_in_shapes,
                                       self._modules_out_shapes,
                                       input_channels=tmp_in_channels,
                                       output_channels=tmp_out_channels,
                                       conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                       linear_op_metatypes=LINEAR_LAYER_METATYPES)

    def _find_uniform_pruning_rate_for_target_flops(self, target_flops_pruning_rate: float) -> float:
        """
        Searching for the minimal uniform layer-wise weight pruning rate (proportion of zero filters in a layer)
         needed to achieve the target pruning rate in flops.

        :param target_flops_pruning_rate: target proportion of flops that should be pruned in the model
        :return: uniform pruning rate for all layers
        """
        error = 0.01
        target_flops = self.full_flops * (1 - target_flops_pruning_rate)
        left, right = 0.0, 1.0
        while abs(right - left) > error:
            middle = (left + right) / 2
            flops, params_num = self._calculate_flops_and_weights_in_uniformly_pruned_model(middle)
            if flops < target_flops:
                right = middle
            else:
                left = middle
        flops, params_num = self._calculate_flops_and_weights_in_uniformly_pruned_model(right)
        if flops < target_flops:
            self.current_flops = flops
            self.current_params_num = params_num
            return right
        raise RuntimeError("Can't prune the model to get the required "
                           "pruning rate in flops = {}".format(target_flops_pruning_rate))

    def set_pruning_rate(self, pruning_rate: Union[float, Dict[int, float]],
                         run_batchnorm_adaptation: bool = False) -> None:
        """
        Set the global or groupwise pruning rate in the model.
        If pruning_rate is a float, the correspoding global pruning rate is set in the model,
        either in terms of the percentage of filters pruned or as the percentage of flops
        removed, the latter being true in case the "prune_flops" flag of the controller is
        set to True.
        If pruning_rate is a dict, the keys should correspond to layer group id's and the
        values to groupwise pruning rates to be set in the model.
        """
        groupwise_pruning_rates_set = isinstance(pruning_rate, dict)
        passed_pruning_rate = pruning_rate

        if not self.frozen:
            nncf_logger.info("Computing filter importance scores and binary masks...")
            with torch.no_grad():
                if self.all_weights:
                    if groupwise_pruning_rates_set:
                        raise RuntimeError('Cannot set group-wise pruning rates with '
                                           'all_weights=True')
                    # Non-uniform (global) importance-score-based pruning according
                    # to the global pruning rate
                    if self.prune_flops:
                        self._set_binary_masks_for_pruned_modules_globally_by_flops_target(pruning_rate)
                    else:
                        self._set_binary_masks_for_pruned_modules_globally(pruning_rate)
                else:
                    if groupwise_pruning_rates_set:
                        group_ids = [group.id for group in self.pruned_module_groups_info.get_all_clusters()]
                        if set(pruning_rate.keys()) != set(group_ids):
                            raise RuntimeError('Groupwise pruning rate dict keys do not correspond to '
                                               'layer group ids')
                    else:
                        # Pruning uniformly with the same pruning rate across layers
                        if self.prune_flops:
                            # Looking for layerwise pruning rate needed for the required flops pruning rate
                            pruning_rate = self._find_uniform_pruning_rate_for_target_flops(pruning_rate)
                    self._set_binary_masks_for_pruned_modules_groupwise(pruning_rate)

        self._propagate_masks()
        if not groupwise_pruning_rates_set:
            self._pruning_rate = passed_pruning_rate
        else:
            self._pruning_rate = self._calculate_global_weight_pruning_rate()

        if run_batchnorm_adaptation:
            self._run_batchnorm_adaptation()

    def _calculate_global_weight_pruning_rate(self) -> float:
        full_param_count = 0
        pruned_param_count = 0
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            layer_param_count = sum(p.numel() for p in minfo.module.parameters() if p.requires_grad)
            layer_weight_pruning_rate = self.pruning_rate_for_mask(minfo)
            full_param_count += layer_param_count
            pruned_param_count += layer_param_count * layer_weight_pruning_rate
        return pruned_param_count / full_param_count

    @property
    def current_groupwise_pruning_rate(self) -> Dict[int, float]:
        """
        Return the dict of layer group id's and corresponding current groupwise
        pruning rates in the model
        """
        groupwise_pruning_rate_dict = {}
        for group in self.pruned_module_groups_info.get_all_clusters():
            groupwise_pruning_rate_dict[group.id] = self.pruning_rate_for_mask(group.elements[0])
        return groupwise_pruning_rate_dict

    def _set_binary_masks_for_pruned_modules_groupwise(self,
                                                       pruning_rate: Union[float, Dict[int, float]]) -> None:
        """
        Set the binary mask values according to groupwise pruning rates.
        If pruning_rate is a float, set the pruning rates uniformly across groups.
        If pruning_rate is a dict, set specific pruning rates corresponding to each group.
        """
        nncf_logger.debug("Updating binary masks for pruned modules.")
        groupwise_pruning_rates_set = isinstance(pruning_rate, dict)

        for group in self.pruned_module_groups_info.get_all_clusters():
            group_pruning_rate = pruning_rate[group.id] if groupwise_pruning_rates_set \
                else pruning_rate

            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in group.elements])
            assert torch.all(filters_num == filters_num[0])
            device = group.elements[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # 1. Calculate cumulative importance for all filters in group
            for minfo in group.elements:
                filters_importance = self.filter_importance(minfo.module.weight,
                                                            minfo.module.target_weight_dim_for_compression)
                cumulative_filters_importance += filters_importance

            # 2. Calculate threshold
            num_of_sparse_elems = get_rounded_pruned_element_number(cumulative_filters_importance.size(0),
                                                                    group_pruning_rate)
            threshold = sorted(cumulative_filters_importance)[min(num_of_sparse_elems, filters_num[0] - 1)]
            mask = calculate_binary_mask(cumulative_filters_importance, threshold)

            # 3. Set binary masks for filter
            for minfo in group.elements:
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = mask

        # Calculate actual flops and weights number with new masks
        self._update_benchmark_statistics()

    def _set_binary_masks_for_pruned_modules_globally(self, pruning_rate: float) -> None:
        """
        Set the binary mask values for layer groups according to the global pruning rate.
        Filter importance scores in each group are merged into a single global list and a
        threshold value separating the pruning_rate proportion of the least important filters
        in the model is calculated. Filters are pruned globally according to the threshold value.
        """
        nncf_logger.debug("Setting new binary masks for all pruned modules together.")
        filter_importances = []
        # 1. Calculate importances for all groups of  filters
        for group in self.pruned_module_groups_info.get_all_clusters():
            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in group.elements])
            assert torch.all(filters_num == filters_num[0])
            device = group.elements[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # Calculate cumulative importance for all filters in this group
            for minfo in group.elements:
                normalized_weight = self.weights_normalizer(minfo.module.weight)
                filters_importance = self.filter_importance(normalized_weight,
                                                            minfo.module.target_weight_dim_for_compression)
                cumulative_filters_importance += filters_importance

            filter_importances.append(cumulative_filters_importance)

        # 2. Calculate one threshold for all weights
        importances = torch.cat(filter_importances)
        threshold = sorted(importances)[int(pruning_rate * importances.size(0))]

        # 3. Set binary masks for filters in groups
        for i, group in enumerate(self.pruned_module_groups_info.get_all_clusters()):
            mask = calculate_binary_mask(filter_importances[i], threshold)
            for minfo in group.elements:
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = mask

        # Calculate actual flops and weights number with new masks
        self._update_benchmark_statistics()

    def _set_binary_masks_for_pruned_modules_globally_by_flops_target(self,
                                                                      target_flops_pruning_rate: float) -> None:
        """
        Sorting all prunable filters in the network by importance and pruning the amount of the
        least important filters sufficient to achieve the target pruning rate by flops.
        Filters are pruned one-by-one and the corresponding flops value is checked.

        :param target_flops_pruning_rate: target proportion of flops removed from the model
        :return:
        """
        target_flops = self.full_flops * (1 - target_flops_pruning_rate)

        # 1. Initialize masks
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            new_mask = torch.ones(get_filters_num(minfo.module)).to(
                minfo.module.weight.device)
            self.set_mask(minfo, new_mask)

        # 2. Calculate filter importances for all prunable groups
        filter_importances = []
        cluster_indexes = []
        filter_indexes = []

        for cluster in self.pruned_module_groups_info.get_all_clusters():
            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in cluster.elements])
            assert torch.all(filters_num == filters_num[0])
            device = cluster.elements[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # Calculate cumulative importance for all filters in this group
            for minfo in cluster.elements:
                weight = minfo.module.weight
                if self.normalize_weights:
                    weight = self.weights_normalizer(weight)
                filters_importance = self.filter_importance(weight,
                                                            minfo.module.target_weight_dim_for_compression)
                scaled_importance = self.ranking_coeffs[minfo.node_name][0] * filters_importance + \
                                    self.ranking_coeffs[minfo.node_name][1]
                cumulative_filters_importance += scaled_importance

            filter_importances.append(cumulative_filters_importance)
            cluster_indexes.append(cluster.id * torch.ones_like(cumulative_filters_importance))
            filter_indexes.append(torch.arange(len(cumulative_filters_importance)))

        importances = torch.cat(filter_importances)
        cluster_indexes = torch.cat(cluster_indexes)
        filter_indexes = torch.cat(filter_indexes)

        # 3. Sort all filter groups by importances and prune the least important filters
        # until target flops pruning rate is achieved
        sorted_importances = sorted(zip(importances, cluster_indexes, filter_indexes), key=lambda x: x[0])
        cur_num = 0
        tmp_in_channels = self._modules_in_channels.copy()
        tmp_out_channels = self._modules_out_channels.copy()
        tmp_pruning_quotas = self.pruning_quotas.copy()

        while cur_num < len(sorted_importances):
            cluster_idx = int(sorted_importances[cur_num][1])
            filter_idx = int(sorted_importances[cur_num][2])

            if tmp_pruning_quotas[cluster_idx] > 0:
                tmp_pruning_quotas[cluster_idx] -= 1
            else:
                cur_num += 1
                continue

            cluster = self.pruned_module_groups_info.get_cluster_by_id(cluster_idx)
            for node in cluster.elements:
                tmp_out_channels[node.node_name] -= 1
                node.operand.binary_filter_pruning_mask[filter_idx] = 0

            # Prune in channels in all next nodes
            next_nodes = self.next_nodes[cluster.id]
            for node_id in next_nodes:
                tmp_in_channels[node_id] -= 1

            flops, params_num = count_flops_and_weights(self._model.get_original_graph(),
                                                        self._modules_in_shapes,
                                                        self._modules_out_shapes,
                                                        input_channels=tmp_in_channels,
                                                        output_channels=tmp_out_channels,
                                                        conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
                                                        linear_op_metatypes=LINEAR_LAYER_METATYPES)
            if flops < target_flops:
                self.current_flops = flops
                self.current_params_num = params_num
                return
            cur_num += 1
        raise RuntimeError("Can't prune model to asked flops pruning rate")

    def _propagate_masks(self):
        nncf_logger.debug("Propagating pruning masks")
        # 1. Propagate masks for all modules
        graph = self.model.get_original_graph()

        init_output_masks_in_graph(graph, self.pruned_module_groups_info.get_all_nodes())
        MaskPropagationAlgorithm(graph, PT_PRUNING_OPERATOR_METATYPES).mask_propagation()

        # 2. Set the masks for Batch/Group Norms
        pruned_node_modules = []
        for node, pruning_block, node_module in self._pruned_norms_operators:
            if node_module not in pruned_node_modules:
                # Setting masks for BN nodes
                pruning_block.binary_filter_pruning_mask = node.data['output_mask'].tensor
                pruned_node_modules.append(node_module)

    def prepare_for_export(self):
        """
        Applies pruning masks to layer weights before exporting the model to ONNX.
        """
        self._propagate_masks()

        pruned_layers_stats = self.get_stats_for_pruned_modules()
        nncf_logger.debug('Pruned layers statistics: \n%s', pruned_layers_stats.draw())

    def compression_stage(self) -> CompressionStage:
        target_pruning_level = self.scheduler.target_level
        actual_pruning_level = self._pruning_rate
        if actual_pruning_level == 0:
            return CompressionStage.UNCOMPRESSED
        if actual_pruning_level >= target_pruning_level:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    @property
    def compression_rate(self):
        if self.prune_flops:
            return 1 - self.current_flops / self.full_flops
        return self.pruning_rate

    @compression_rate.setter
    def compression_rate(self, pruning_rate):
        is_pruning_controller_frozen = self.frozen
        self.freeze(False)
        self.set_pruning_rate(pruning_rate)
        self.freeze(is_pruning_controller_frozen)

    def disable_scheduler(self):
        self._scheduler = StubCompressionScheduler()
        self._scheduler.current_pruning_level = 0.0

    def _update_benchmark_statistics(self):
        self.current_flops, self.current_params_num = self._calculate_flops_and_weights_pruned_model_by_masks()

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(**extract_bn_adaptation_init_params(self.config,
                                                                                                   'filter_pruning'))
        self._bn_adaptation.run(self.model)
