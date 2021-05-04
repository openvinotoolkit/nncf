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

from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch
from texttable import Texttable
from torch import nn

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.model_analysis import Clusterization
from nncf.common.pruning.utils import calculate_in_out_channels_in_uniformly_pruned_model
from nncf.common.pruning.utils import count_flops_for_nodes
from nncf.common.pruning.utils import get_cluster_next_nodes
from nncf.common.pruning.utils import get_conv_in_out_channels
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.common.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.common.utils.logger import logger as nncf_logger
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.dynamic_graph.context import Scope
from nncf.layers import NNCF_PRUNING_MODULES_DICT
from nncf.layers import NNCF_GENERAL_CONV_MODULES_DICT
from nncf.layer_utils import _NNCFModuleMixin
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoBuilder
from nncf.pruning.base_algo import PrunedModuleInfo
from nncf.pruning.base_algo import BasePruningAlgoController
from nncf.pruning.export_helpers import ModelPruner
from nncf.pruning.export_helpers import PTElementwise
from nncf.pruning.export_helpers import PT_PRUNING_OPERATOR_METATYPES
from nncf.pruning.filter_pruning.functions import calculate_binary_mask
from nncf.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from nncf.pruning.filter_pruning.functions import tensor_l2_normalizer
from nncf.pruning.filter_pruning.layers import FilterPruningBlock
from nncf.pruning.filter_pruning.layers import inplace_apply_filter_binary_mask
from nncf.pruning.utils import init_output_masks_in_graph
from nncf.utils import get_filters_num


@COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    def create_weight_pruning_operation(self, module):
        return FilterPruningBlock(module.weight.size(module.target_weight_dim_for_compression))

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return FilterPruningController(target_model,
                                       self._prunable_types,
                                       self.pruned_module_groups_info,
                                       self.config)

    def _is_pruned_module(self, module) -> bool:
        # Currently prune only Convolutions
        return isinstance(module, tuple(NNCF_PRUNING_MODULES_DICT.keys()))

    def get_op_types_of_pruned_modules(self) -> List[str]:
        types = [v.op_func_name for v in NNCF_PRUNING_MODULES_DICT]
        return types

    def get_types_of_grouping_ops(self) -> List[str]:
        return PTElementwise.get_all_op_aliases()


class FilterPruningController(BasePruningAlgoController):
    def __init__(self, target_model: NNCFNetwork,
                 prunable_types: List[str],
                 pruned_module_groups: Clusterization,
                 config):
        super().__init__(target_model, prunable_types, pruned_module_groups, config)
        params = self.config.get("params", {})
        self.frozen = False
        self._pruning_rate = 0
        self.pruning_init = config.get("pruning_init", 0)
        self.pruning_quota = 1.0

        self.modules_in_channels = {}  # type: Dict[Scope, int]
        self.modules_out_channels = {}  # type: Dict[Scope, int]
        self.pruning_quotas = {}
        self.nodes_flops = {}  # type: Dict[Scope, int]
        self.nodes_flops_cost = {}  # type: Dict[Scope, int]
        self.next_nodes = {}
        self._init_pruned_modules_params()
        self.flops_count_init()
        self.full_flops = sum(self.nodes_flops.values())
        self.current_flops = self.full_flops

        self.weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self.filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(params.get('weight_importance', 'L2'))
        self.all_weights = params.get("all_weights", False)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get("schedule", "baseline"))

        self.set_pruning_rate(self.pruning_init)
        self._scheduler = scheduler_cls(self, params)

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @staticmethod
    def _get_mask(minfo: PrunedModuleInfo):
        return minfo.operand.binary_filter_pruning_mask

    def statistics(self, quickly_collected_only=False):
        stats = super().statistics(quickly_collected_only)
        stats['pruning_rate'] = self._pruning_rate
        stats['FLOPS pruning level'] = 1 - self.current_flops / self.full_flops
        stats['FLOPS current / full'] = f"{self.current_flops} / {self.full_flops}"
        return stats

    @property
    def pruning_rate(self) -> float:
        """Global pruning rate in the model"""
        return self._pruning_rate

    def freeze(self):
        self.frozen = True

    def step(self, next_step):
        self._apply_masks()

    def _init_pruned_modules_params(self):
        # 1. Init in/out channels for potentially prunable modules
        graph = self._model.get_original_graph()
        self.modules_in_channels, self.modules_out_channels = get_conv_in_out_channels(graph)

        # 2. Init next_nodes for every pruning cluster
        self.next_nodes = get_cluster_next_nodes(graph, self.pruned_module_groups_info, self._prunable_types)

        # 3. Init pruning quotas
        for cluster in self.pruned_module_groups_info.get_all_clusters():
            self.pruning_quotas[cluster.id] = self.modules_out_channels[cluster.nodes[0].module_scope] \
                                              * self.pruning_quota

    def flops_count_init(self) -> None:
        def get_in_out_shapes_hook(in_shapes_dict_to_save: dict, out_shapes_dict_to_save: dict):
            ctx = self._model.get_tracing_context()

            def compute_in_out_shapes_hook(module, input_, output):
                if isinstance(module, tuple(NNCF_GENERAL_CONV_MODULES_DICT.values())):
                    out_shapes_dict_to_save[ctx.scope] = output.shape[2:]
                if isinstance(module, nn.Linear):
                    out_shapes_dict_to_save[ctx.scope] = output.shape[-1]
                    if len(input_[0].shape) == 1:
                        in_shapes_dict_to_save[ctx.scope] = input_[0].shape[0]
                    else:
                        in_shapes_dict_to_save[ctx.scope] = input_[0].shape[1:]

            return compute_in_out_shapes_hook

        def get_node_cost_hook():
            """
            Cost of node is num of flops for this node divided by numbers of input and output channels for this node.
            """
            ctx = self._model.get_tracing_context()

            def compute_cost_hook(module, input_, output):
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                                       nn.ConvTranspose3d)):
                    ks = module.weight.data.shape
                    cost = 2 * np.prod(ks[2:]) * np.prod(output.shape[2:]) / module.groups
                else:
                    return
                self.nodes_flops_cost[ctx.scope] = cost

            return compute_cost_hook

        graph = self._model.get_original_graph()
        hook_list = []
        in_shapes, out_shapes = {}, {}

        for nncf_node in graph.get_all_nodes():
            node_module = self._model.get_module_by_scope(nncf_node.ia_op_exec_context.scope_in_model)
            hook_list.append(node_module.register_forward_hook(get_in_out_shapes_hook(in_shapes, out_shapes)))
            hook_list.append(node_module.register_forward_hook(get_node_cost_hook()))

        self._model.do_dummy_forward(force_eval=True)

        self.nodes_flops = count_flops_for_nodes(graph, in_shapes, out_shapes,
                                                 conv_op_types=[v.op_func_name
                                                                for v in NNCF_GENERAL_CONV_MODULES_DICT],
                                                 linear_op_types=['linear'])
        for h in hook_list:
            h.remove()

    def _calculate_flops_pruned_model_by_masks(self) -> float:
        """
        Calculates number of flops for pruned model by using binary_filter_pruning_mask.
        :return: number of flops in model
        """
        tmp_in_channels = self.modules_in_channels.copy()
        tmp_out_channels = self.modules_out_channels.copy()

        for group in self.pruned_module_groups_info.get_all_clusters():
            assert all(tmp_out_channels[group.nodes[0].module_scope] == tmp_out_channels[node.module_scope] for node in
                       group.nodes)
            new_out_channels_num = int(sum(group.nodes[0].operand.binary_filter_pruning_mask))
            num_of_sparse_elems = len(group.nodes[0].operand.binary_filter_pruning_mask) - new_out_channels_num
            for node in group.nodes:
                tmp_out_channels[node.module_scope] = new_out_channels_num
            # Prune in_channels in all next nodes of cluster
            next_nodes = self.next_nodes[group.id]
            for node_module_scope in next_nodes:
                tmp_in_channels[node_module_scope] -= num_of_sparse_elems

        flops = self._calculate_flops_in_pruned_model(tmp_in_channels, tmp_out_channels)
        return flops

    def _calculate_flops_in_pruned_model(self, modules_in_channels: Dict[Scope, int],
                                         modules_out_channels: Dict[Scope, int]) -> float:
        """
        Calculates number of flops in model with number of input/output channels for nodes from modules_in_channels,
        modules_out_channels. It allows to count the number of flops in pruned model (with changed number of
        input/output channels for some nodes).
        Number of flops calculates as follows: for nodes that isn't in in/out channels used full number of flops from
        self.nodes_flops.
        For nodes with keys from in/out channels dicts flops
        = modules_in_channels[node_id] * modules_out_channels[node_id] * self.nodes_flops_cost[node_id]
        :param modules_in_channels: numbers of input channels in nodes
        :param modules_out_channels: numbers of output channels in model
        :return: number of flops in model
        """
        flops = 0
        graph = self._model.get_original_graph()
        for nncf_node in graph.get_all_nodes():
            scope = nncf_node.ia_op_exec_context.scope_in_model
            if scope in modules_in_channels:
                flops += int(modules_in_channels[scope] * modules_out_channels[scope] * \
                         self.nodes_flops_cost[scope])
            elif scope in self.nodes_flops:
                flops += self.nodes_flops[scope]
        return flops

    def _calculate_flops_in_uniformly_pruned_model(self, pruning_rate: float) -> float:
        """
        Prune all prunable modules in model with pruning_rate rate and returns flops of pruned model.
        :param pruning_rate: proportion of zero filters in all modules
        :return: flops number in pruned model
        """
        tmp_in_channels, tmp_out_channels = \
            calculate_in_out_channels_in_uniformly_pruned_model(
                pruning_groups=self.pruned_module_groups_info.get_all_clusters(),
                pruning_rate=pruning_rate,
                full_input_channels=self.modules_in_channels,
                full_output_channels=self.modules_out_channels,
                pruning_groups_next_nodes=self.next_nodes)
        flops = self._calculate_flops_in_pruned_model(tmp_in_channels, tmp_out_channels)
        return flops

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
            flops = self._calculate_flops_in_uniformly_pruned_model(middle)
            if flops < target_flops:
                right = middle
            else:
                left = middle
        flops = self._calculate_flops_in_uniformly_pruned_model(right)
        if flops < target_flops:
            self.current_flops = flops
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

            if self.zero_grad:
                self.zero_grads_for_pruned_modules()

        self._apply_masks()

        if not groupwise_pruning_rates_set:
            self._pruning_rate = passed_pruning_rate
        else:
            self._pruning_rate = self._calculate_global_weight_pruning_rate()

        if run_batchnorm_adaptation:
            self.run_batchnorm_adaptation(self.config)

    def _calculate_global_weight_pruning_rate(self) -> float:
        full_param_count = 0
        pruned_param_count = 0
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            layer_param_count = sum(p.numel() for p in minfo.module.parameters() if p.requires_grad)
            layer_weight_pruning_rate = self.pruning_rate_for_weight(minfo)
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
            groupwise_pruning_rate_dict[group.id] = self.pruning_rate_for_mask(group.nodes[0])
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

            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in group.nodes])
            assert torch.all(filters_num == filters_num[0])
            device = group.nodes[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # 1. Calculate cumulative importance for all filters in group
            for minfo in group.nodes:
                filters_importance = self.filter_importance(minfo.module.weight,
                                                            minfo.module.target_weight_dim_for_compression)
                cumulative_filters_importance += filters_importance

            # 2. Calculate threshold
            num_of_sparse_elems = get_rounded_pruned_element_number(cumulative_filters_importance.size(0),
                                                                    group_pruning_rate)
            threshold = sorted(cumulative_filters_importance)[min(num_of_sparse_elems, filters_num[0] - 1)]
            mask = calculate_binary_mask(cumulative_filters_importance, threshold)

            # 3. Set binary masks for filter
            for minfo in group.nodes:
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = mask

        # Calculate actual flops with new masks
        self.current_flops = self._calculate_flops_pruned_model_by_masks()

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
            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in group.nodes])
            assert torch.all(filters_num == filters_num[0])
            device = group.nodes[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # Calculate cumulative importance for all filters in this group
            for minfo in group.nodes:
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
            for minfo in group.nodes:
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = mask

        # Calculate actual flops with new masks
        self.current_flops = self._calculate_flops_pruned_model_by_masks()

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
            pruning_module = minfo.operand
            pruning_module.binary_filter_pruning_mask = torch.ones(get_filters_num(minfo.module)).to(
                minfo.module.weight.device)

        # 2. Calculate filter importances for all prunable groups
        filter_importances = []
        cluster_indexes = []
        filter_indexes = []

        for cluster in self.pruned_module_groups_info.get_all_clusters():
            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in cluster.nodes])
            assert torch.all(filters_num == filters_num[0])
            device = cluster.nodes[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # Calculate cumulative importance for all filters in this group
            for minfo in cluster.nodes:
                normalized_weight = self.weights_normalizer(minfo.module.weight)
                filters_importance = self.filter_importance(normalized_weight,
                                                            minfo.module.target_weight_dim_for_compression)
                cumulative_filters_importance += filters_importance

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
        tmp_in_channels = self.modules_in_channels.copy()
        tmp_out_channels = self.modules_out_channels.copy()
        while cur_num < len(sorted_importances):
            cluster_idx = int(sorted_importances[cur_num][1])
            filter_idx = int(sorted_importances[cur_num][2])

            if self.pruning_quotas[cluster_idx] > 0:
                self.pruning_quotas[cluster_idx] -= 1
            else:
                cur_num += 1
                continue

            cluster = self.pruned_module_groups_info.get_cluster_by_id(cluster_idx)
            for node in cluster.nodes:
                tmp_out_channels[node.module_scope] -= 1
                node.operand.binary_filter_pruning_mask[filter_idx] = 0

            # Prune in channels in all next nodes
            next_nodes = self.next_nodes[cluster.id]
            for node_id in next_nodes:
                tmp_in_channels[node_id] -= 1

            flops = self._calculate_flops_in_pruned_model(tmp_in_channels, tmp_out_channels)
            if flops < target_flops:
                self.current_flops = flops
                return
            cur_num += 1
        raise RuntimeError("Can't prune model to asked flops pruning rate")

    def _apply_masks(self):
        nncf_logger.debug("Applying pruning binary masks")

        def _apply_binary_mask_to_module_weight_and_bias(module, mask, module_scope):
            with torch.no_grad():
                dim = module.target_weight_dim_for_compression if isinstance(module, _NNCFModuleMixin) else 0
                # Applying the mask to weights
                inplace_apply_filter_binary_mask(mask, module.weight, module_scope, dim)
                # Applying the mask to biases (if they exist)
                if module.bias is not None:
                    inplace_apply_filter_binary_mask(mask, module.bias, module_scope)

        # 1. Propagate masks for all modules
        graph = self.model.get_original_graph()

        init_output_masks_in_graph(graph, self.pruned_module_groups_info.get_all_nodes())
        MaskPropagationAlgorithm(graph, PT_PRUNING_OPERATOR_METATYPES).mask_propagation()

        # 2. Apply the masks
        types_to_apply_mask = [v.op_func_name for v in NNCF_GENERAL_CONV_MODULES_DICT] + ['group_norm']
        if self.prune_batch_norms:
            types_to_apply_mask.append('batch_norm')

        pruned_node_modules = list()
        for node in graph.get_all_nodes():
            if node.node_type not in types_to_apply_mask:
                continue
            scope = node.ia_op_exec_context.scope_in_model
            node_module = self.model.get_module_by_scope(scope)
            if node.data['output_mask'] is not None and node_module not in pruned_node_modules:
                _apply_binary_mask_to_module_weight_and_bias(node_module, node.data['output_mask'], scope)
                pruned_node_modules.append(node_module)

    @staticmethod
    def create_stats_table_for_pruning_export(old_modules_info, new_modules_info):
        """
        Creating a table with comparison of model params number before and after pruning.
        :param old_modules_info: Information about pruned layers before actually prune layers.
        :param new_modules_info: Information about pruned layers after actually prune layers.
        """
        table = Texttable()
        header = ["Name", "Weight's shape", "New weight's shape", "Bias shape", "New bias shape",
                  "Weight's params count", "New weight's params count",
                  "Mask zero %", "Layer PR"]
        data = [header]

        for layer in old_modules_info.keys():
            assert layer in new_modules_info

            drow = {h: 0 for h in header}
            drow["Name"] = layer
            drow["Weight's shape"] = old_modules_info[layer]['w_shape']
            drow["New weight's shape"] = new_modules_info[layer]['w_shape']
            drow["Bias shape"] = old_modules_info[layer]['b_shape']
            drow["New bias shape"] = new_modules_info[layer]['b_shape']

            drow["Weight's params count"] = old_modules_info[layer]['params_count']
            drow["New weight's params count"] = new_modules_info[layer]['params_count']

            drow["Mask zero %"] = old_modules_info[layer]['mask_pr']

            drow["Layer PR"] = 1 - new_modules_info[layer]['params_count'] / old_modules_info[layer]['params_count']
            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)
        return table

    def prepare_for_export(self):
        """
        This function discards the pruned filters based on the binary masks
        before exporting the model to ONNX.
        """
        self._apply_masks()
        model = self._model.eval().cpu()
        graph = model.get_original_graph()

        parameters_count_before = model.get_parameters_count_in_model()
        flops = model.get_MACs_in_model()
        pruned_layers_stats = self.get_stats_for_pruned_modules()

        init_output_masks_in_graph(graph, self.pruned_module_groups_info.get_all_nodes())
        model_pruner = ModelPruner(model, graph, PT_PRUNING_OPERATOR_METATYPES)
        model_pruner.prune_model()

        parameters_count_after = model.get_parameters_count_in_model()
        flops_after = model.get_MACs_in_model()
        new_pruned_layers_stats = self.get_stats_for_pruned_modules()
        stats = self.create_stats_table_for_pruning_export(pruned_layers_stats, new_pruned_layers_stats)

        nncf_logger.info(stats.draw())
        nncf_logger.info('Final Model Pruning Rate = %.3f', 1 - parameters_count_after / parameters_count_before)
        nncf_logger.info('Total MAC pruning level = %.3f', 1 - flops_after / flops)

    def compression_level(self) -> CompressionLevel:
        target_pruning_level = self.scheduler.target_level
        actual_pruning_level = self._pruning_rate
        if actual_pruning_level == 0:
            return CompressionLevel.NONE
        if actual_pruning_level >= target_pruning_level:
            return CompressionLevel.FULL
        return CompressionLevel.PARTIAL
