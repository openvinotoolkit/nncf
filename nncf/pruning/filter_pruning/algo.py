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
from texttable import Texttable

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmController, CompressionLevel
from nncf.layers import NNCF_PRUNING_MODULES_DICT
from nncf.layer_utils import _NNCFModuleMixin
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoBuilder, PrunedModuleInfo, BasePruningAlgoController
from nncf.pruning.export_helpers import ModelPruner, Elementwise
from nncf.pruning.filter_pruning.functions import calculate_binary_mask, FILTER_IMPORTANCE_FUNCTIONS, \
    tensor_l2_normalizer
from nncf.pruning.filter_pruning.layers import FilterPruningBlock, inplace_apply_filter_binary_mask
from nncf.pruning.model_analysis import Clusterization
from nncf.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.pruning.utils import get_rounded_pruned_element_number
from nncf.utils import get_filters_num


@COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    def create_weight_pruning_operation(self, module):
        return FilterPruningBlock(module.weight.size(module.target_compression_weight_dim))

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return FilterPruningController(target_model,
                                       self.pruned_module_groups_info,
                                       self.config)

    def _is_pruned_module(self, module):
        # Currently prune only Convolutions
        return isinstance(module, tuple(NNCF_PRUNING_MODULES_DICT.keys()))

    def get_types_of_pruned_modules(self):
        types = [str.lower(v.__name__) for v in NNCF_PRUNING_MODULES_DICT.values()]
        return types + ["conv_transpose2d", "conv_transpose3d"]

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

        self.weights_normalizer = tensor_l2_normalizer  # for all weights in common case
        self.filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(params.get('weight_importance', 'L2'))
        self.all_weights = params.get("all_weights", False)
        scheduler_cls = PRUNING_SCHEDULERS.get(params.get("schedule", "baseline"))
        self._scheduler = scheduler_cls(self, params)

    @staticmethod
    def _get_mask(minfo: PrunedModuleInfo):
        return minfo.operand.binary_filter_pruning_mask

    def statistics(self):
        stats = super().statistics()
        stats['pruning_rate'] = self.pruning_rate
        return stats

    def freeze(self):
        self.frozen = True

    def set_pruning_rate(self, pruning_rate):
        self.pruning_rate = pruning_rate
        if not self.frozen:
            if self.all_weights:
                self._set_binary_masks_for_all_pruned_modules()
            else:
                self._set_binary_masks_for_filters()
            if self.zero_grad:
                self.zero_grads_for_pruned_modules()
        self._apply_masks()
        self.run_batchnorm_adaptation(self.config)

    def _set_binary_masks_for_filters(self):
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
                                                                minfo.module.target_compression_weight_dim)
                    cumulative_filters_importance += filters_importance

                # 2. Calculate threshold
                num_of_sparse_elems = get_rounded_pruned_element_number(cumulative_filters_importance.size(0),
                                                                        self.pruning_rate)
                threshold = sorted(cumulative_filters_importance)[num_of_sparse_elems]
                mask = calculate_binary_mask(cumulative_filters_importance, threshold)

                # 3. Set binary masks for filter
                for minfo in group.nodes:
                    pruning_module = minfo.operand
                    pruning_module.binary_filter_pruning_mask = mask

    def _set_binary_masks_for_all_pruned_modules(self):
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
                                                            minfo.module.target_compression_weight_dim)
                cumulative_filters_importance += filters_importance

            filter_importances.append(cumulative_filters_importance)

        # 2. Calculate one threshold for all weights
        importances = torch.cat(filter_importances)
        threshold = sorted(importances)[int(self.pruning_rate * importances.size(0))]

        # 3. Set binary masks for filters in grops
        for i, group in enumerate(self.pruned_module_groups_info.get_all_clusters()):
            mask = calculate_binary_mask(filter_importances[i], threshold)
            for minfo in group.nodes:
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = mask

    def _apply_masks(self):
        nncf_logger.debug("Applying pruning binary masks")

        def _apply_binary_mask_to_module_weight_and_bias(module, mask, module_name=""):
            with torch.no_grad():
                dim = module.target_compression_weight_dim if isinstance(module, _NNCFModuleMixin) else 0
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
                    and related_modules[PrunedModuleInfo.BN_MODULE_NAME] is not None:
                bn_module = related_modules[PrunedModuleInfo.BN_MODULE_NAME]
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
