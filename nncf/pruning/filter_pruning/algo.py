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
import numpy as np
import torch
from functools import partial
from texttable import Texttable
from torch import nn

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmController, CompressionLevel
from nncf.layers import NNCF_PRUNING_MODULES_DICT
from nncf.layer_utils import _NNCFModuleMixin
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoBuilder, PrunedModuleInfo, BasePruningAlgoController
from nncf.pruning.export_helpers import ModelPruner, Elementwise, Convolution
from nncf.pruning.filter_pruning.functions import calculate_binary_mask, FILTER_IMPORTANCE_FUNCTIONS, \
    tensor_l2_normalizer
from nncf.pruning.filter_pruning.layers import FilterPruningBlock, inplace_apply_filter_binary_mask
from nncf.pruning.model_analysis import Clusterization
from nncf.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.utils import get_filters_num
from nncf.pruning.utils import get_rounded_pruned_element_number, get_next_nodes_of_types
from nncf.utils import compute_FLOPs_hook


@COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    def create_weight_pruning_operation(self, module):
        return FilterPruningBlock(module.weight.size(module.target_weight_dim_for_compression))

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return FilterPruningController(target_model,
                                       self.pruned_module_groups_info,
                                       self.config)

    def _is_pruned_module(self, module):
        # Currently prune only Convolutions
        return isinstance(module, tuple(NNCF_PRUNING_MODULES_DICT.keys()))

    def get_op_types_of_pruned_modules(self):
        types = [v.op_func_name for v in NNCF_PRUNING_MODULES_DICT]
        return types

    def get_types_of_grouping_ops(self):
        return Elementwise.get_all_op_aliases()


class FilterPruningController(BasePruningAlgoController):
    def __init__(self, target_model: NNCFNetwork,
                 pruned_module_groups: Clusterization,
                 config):
        super().__init__(target_model, pruned_module_groups, config)
        params = self.config.get("params", {})
        self.frozen = False
        self.pruning_rate = 0
        self.pruning_init = config.get("pruning_init", 0)
        self.pruning_quota = 1.0

        if self.prune_flops:
            self.modules_in_channels = {}
            self.modules_out_channels = {}
            self.pruning_quotas = {}
            self.nodes_flops = {}
            self.nodes_flops_cost = {}
            self.next_nodes = {}
            self._init_pruned_modules_params()
            self.flops_count_init()
            self.full_flops = sum(self.nodes_flops.values())

        self.weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self.filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(params.get('weight_importance', 'L2'))
        self.all_weights = params.get("all_weights", False)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get("schedule", "baseline"))
        self.set_pruning_rate(self.pruning_init)
        self._scheduler = scheduler_cls(self, params)


    @staticmethod
    def _get_mask(minfo: PrunedModuleInfo):
        return minfo.operand.binary_filter_pruning_mask

    def statistics(self, quickly_collected_only=False):
        stats = super().statistics()
        stats['pruning_rate'] = self.pruning_rate
        return stats

    def freeze(self):
        self.frozen = True

    def _init_pruned_modules_params(self):
        def get_in_out_channels(module):
            in_channels, out_channels = None, None
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                                   nn.ConvTranspose3d)):
                in_channels = module.in_channels
                out_channels = module.out_channels
            return in_channels, out_channels

        # 1. Init in/out channels for potentially prunable modules
        graph = self._model.get_original_graph()
        for nncf_node in graph.get_all_nodes():
            node_module = self._model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
            in_channels, out_channels = get_in_out_channels(node_module)
            if in_channels:
                self.modules_in_channels[nncf_node.node_id] = in_channels
            if out_channels:
                self.modules_out_channels[nncf_node.node_id] = out_channels

        prunable_types = Convolution.get_all_op_aliases()
        # 2. Init next_nodes for every pruning cluster
        for cluster in self.pruned_module_groups_info.get_all_clusters():
            next_nodes_cluster = set()
            for cluster_node in cluster.nodes:
                nncf_cluster_node = graph.get_nncf_node_by_id(cluster_node.nncf_node_id)
                next_nodes = get_next_nodes_of_types(self._model, nncf_cluster_node, prunable_types)

                next_nodes_idxs = [n.node_id for n in next_nodes]
                next_nodes_cluster = next_nodes_cluster.union(next_nodes_idxs)
            self.next_nodes[cluster.id] = list(next_nodes_cluster - {n.nncf_node_id for n in cluster.nodes})

            self.pruning_quotas[cluster.id] = self.modules_out_channels[cluster.nodes[0].nncf_node_id] \
                                              * self.pruning_quota

    def flops_count_init(self):
        def get_node_flops_hook(name, dict_to_save):
            return partial(compute_FLOPs_hook, dict_to_save=dict_to_save, name=name)

        def get_node_cost_hook(name):
            """
            Cost of node is num of flops for this node divided by numbers of input and output channels for this node.
            """

            def compute_cost_hook(module, input_, output):
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                                       nn.ConvTranspose3d)):
                    ks = module.weight.data.shape
                    cost = 2 * np.prod(ks[2:]) * np.prod(output.shape[2:]) / module.groups
                else:
                    return
                self.nodes_flops_cost[name] = cost

            return compute_cost_hook

        graph = self._model.get_original_graph()
        hook_list = []

        for nncf_node in graph.get_all_nodes():
            node_module = self._model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
            hook_list.append(node_module.register_forward_hook(get_node_flops_hook(nncf_node.node_id,
                                                                                   self.nodes_flops)))
            hook_list.append(node_module.register_forward_hook(get_node_cost_hook(nncf_node.node_id)))

        self._model.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()

    def _calculate_flops_in_pruned_model(self, modules_in_channels, modules_out_channels):
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
            if nncf_node.node_id in modules_in_channels:
                flops += int(modules_in_channels[nncf_node.node_id] * modules_out_channels[nncf_node.node_id] * \
                         self.nodes_flops_cost[nncf_node.node_id])
            elif nncf_node.node_id in self.nodes_flops:
                flops += self.nodes_flops[nncf_node.node_id]
        return flops

    def _calculate_flops_in_uniformly_pruned_model(self, pruning_rate):
        """
        Prune all prunable modules in model with pruning_rate rate and returns flops of pruned model.
        :param pruning_rate: proportion of zero filters in all modules
        :return: flops number in pruned model
        """
        tmp_in_channels = self.modules_in_channels.copy()
        tmp_out_channels = self.modules_out_channels.copy()

        for group in self.pruned_module_groups_info.get_all_clusters():
            assert all([tmp_out_channels[group.nodes[0].nncf_node_id] == tmp_out_channels[node.nncf_node_id] for node in
                        group.nodes])
            # prune all nodes in cluster (by output channels)
            old_out_channels = self.modules_out_channels[group.nodes[0].nncf_node_id]
            num_of_sparse_elems = get_rounded_pruned_element_number(old_out_channels, pruning_rate)
            new_out_channels_num = old_out_channels - num_of_sparse_elems

            for node in group.nodes:
                tmp_out_channels[node.nncf_node_id] = new_out_channels_num

            # Prune in_channels in all next nodes of cluster
            next_nodes = self.next_nodes[group.id]
            for node_id in next_nodes:
                tmp_in_channels[node_id] = new_out_channels_num
        flops = self._calculate_flops_in_pruned_model(tmp_in_channels, tmp_out_channels)
        return flops

    def _find_layerwise_pruning_rate(self, target_flops_pruning_rate):
        """
        Searching for minimal layer-wise pruning rate (proportion of zero filters in a layer, same for all layers)
         needed to achieve target flops pruning rate.
        :param target_flops_pruning_rate: target proportion of flops that should be pruned in the model
        :return: pruning rate for all layers
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
            return right
        raise RuntimeError("Can't prune model to asked flops pruning rate = {}".format(target_flops_pruning_rate))

    def set_pruning_rate(self, pruning_rate, run_batchnorm_adaptation=False):
        # Pruning rate from scheduler can be flops pruning rate or percentage of params that should be pruned
        self.pruning_rate = pruning_rate
        if not self.frozen:
            if self.all_weights:
                if self.prune_flops:
                    self._set_binary_masks_for_all_pruned_modules_by_flops_target(pruning_rate)
                else:
                    self._set_binary_masks_for_all_pruned_modules(pruning_rate)
            else:
                layerwise_pruning_rate = pruning_rate
                if self.prune_flops:
                    # Looking for layerwise pruning rate needed for asked flops pruning rate
                    layerwise_pruning_rate = self._find_layerwise_pruning_rate(pruning_rate)
                self._set_binary_masks_for_filters(layerwise_pruning_rate)

            if self.zero_grad:
                self.zero_grads_for_pruned_modules()
        self._apply_masks()
        if run_batchnorm_adaptation:
            self.run_batchnorm_adaptation(self.config)

    def _set_binary_masks_for_filters(self, pruning_rate):
        nncf_logger.debug("Setting new binary masks for pruned modules.")

        with torch.no_grad():
            for group in self.pruned_module_groups_info.get_all_clusters():
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
                                                                        pruning_rate)
                threshold = sorted(cumulative_filters_importance)[min(num_of_sparse_elems, filters_num[0] - 1)]
                mask = calculate_binary_mask(cumulative_filters_importance, threshold)

                # 3. Set binary masks for filter
                for minfo in group.nodes:
                    pruning_module = minfo.operand
                    pruning_module.binary_filter_pruning_mask = mask

    def _set_binary_masks_for_all_pruned_modules(self, pruning_rate):
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

        # 3. Set binary masks for filters in grops
        for i, group in enumerate(self.pruned_module_groups_info.get_all_clusters()):
            mask = calculate_binary_mask(filter_importances[i], threshold)
            for minfo in group.nodes:
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = mask

    def _set_binary_masks_for_all_pruned_modules_by_flops_target(self, target_flops_pruning_rate):
        """
        Sorting all prunable filters in the network by importance and prune such amount less important filters
        to archieve target pruning rate by flops.
        :param target_flops_pruning_rate: target proportion of flops removed from the model
        :return:
        """
        target_flops = self.full_flops * (1 - target_flops_pruning_rate)

        # 1. Init masks
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            with torch.no_grad():
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

        # 3. Sort all filters groups by importances and prune less important filters while target flops pruning
        # rate is not achieved
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
                tmp_out_channels[node.nncf_node_id] -= 1
                node.operand.binary_filter_pruning_mask[filter_idx] = 0

            # Prune in channels in all next nodes
            next_nodes = self.next_nodes[cluster.id]
            for node_id in next_nodes:
                tmp_in_channels[node_id] -= 1

            flops = self._calculate_flops_in_pruned_model(tmp_in_channels, tmp_out_channels)
            if flops < target_flops:
                return
            cur_num += 1
        raise RuntimeError("Can't prune model to asked flops pruning rate")

    def _apply_masks(self):
        nncf_logger.debug("Applying pruning binary masks")

        def _apply_binary_mask_to_module_weight_and_bias(module, mask, module_name=""):
            with torch.no_grad():
                dim = module.target_weight_dim_for_compression if isinstance(module, _NNCFModuleMixin) else 0
                # Applying mask to weights
                inplace_apply_filter_binary_mask(mask, module.weight, module_name, dim)
                # Applying mask to bias too (if exists)
                if module.bias is not None:
                    inplace_apply_filter_binary_mask(mask, module.bias, module_name)

        for minfo in self.pruned_module_groups_info.get_all_nodes():
            _apply_binary_mask_to_module_weight_and_bias(minfo.module, minfo.operand.binary_filter_pruning_mask,
                                                         minfo.module_name)

            # Applying mask to the BatchNorm node
            related_modules = minfo.related_modules
            if minfo.related_modules is not None and PrunedModuleInfo.BN_MODULE_NAME in minfo.related_modules \
                    and related_modules[PrunedModuleInfo.BN_MODULE_NAME].module is not None:
                bn_module = related_modules[PrunedModuleInfo.BN_MODULE_NAME].module
                _apply_binary_mask_to_module_weight_and_bias(bn_module, minfo.operand.binary_filter_pruning_mask)

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
        # pylint: disable=protected-access
        nx_graph = graph._nx_graph

        parameters_count_before = model.get_parameters_count_in_model()
        flops = model.get_MACs_in_model()
        pruned_layers_stats = self.get_stats_for_pruned_modules()

        model_pruner = ModelPruner(model, graph, nx_graph)
        model_pruner.prune_model()

        parameters_count_after = model.get_parameters_count_in_model()
        flops_after = model.get_MACs_in_model()
        new_pruned_layers_stats = self.get_stats_for_pruned_modules()
        stats = self.create_stats_table_for_pruning_export(pruned_layers_stats, new_pruned_layers_stats)

        nncf_logger.info(stats.draw())
        nncf_logger.info('Final Model Pruning Rate = %.3f', 1 - parameters_count_after / parameters_count_before)
        nncf_logger.info('Total MAC pruning level = %.3f', 1 - flops_after / flops)

    def compression_level(self) -> CompressionLevel:
        target_pruning_level = self.scheduler.pruning_target
        actual_pruning_level = self.pruning_rate
        if actual_pruning_level == 0:
            return CompressionLevel.NONE
        if actual_pruning_level >= target_pruning_level:
            return CompressionLevel.FULL
        return CompressionLevel.PARTIAL
