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
from typing import List

import torch
from texttable import Texttable

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmController, CompressionLevel
from nncf.layers import NNCF_CONV_MODULES_DICT
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoBuilder, PrunedModuleInfo, BasePruningAlgoController
from nncf.pruning.export_helpers import ModelPruner
from nncf.pruning.filter_pruning.functions import calculate_binary_mask, FILTER_IMPORTANCE_FUNCTIONS, \
    tensor_l2_normalizer
from nncf.pruning.filter_pruning.layers import FilterPruningBlock, inplace_apply_filter_binary_mask
from nncf.pruning.schedulers import PRUNING_SCHEDULERS
from nncf.pruning.utils import get_rounded_pruned_element_number


@COMPRESSION_ALGORITHMS.register('filter_pruning')
class FilterPruningBuilder(BasePruningAlgoBuilder):
    def create_weight_pruning_operation(self, module):
        return FilterPruningBlock(module.weight.size(0))

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return FilterPruningController(target_model,
                                       self._pruned_module_info,
                                       self.config)

    def _is_pruned_module(self, module):
        # Currently prune only Convolutions
        return isinstance(module, tuple(NNCF_CONV_MODULES_DICT.keys()))

    def get_types_of_pruned_modules(self):
        types = [str.lower(v.__name__) for v in NNCF_CONV_MODULES_DICT.values()]
        return types


class FilterPruningController(BasePruningAlgoController):
    def __init__(self, target_model: NNCFNetwork,
                 pruned_module_info: List[PrunedModuleInfo],
                 config):
        super().__init__(target_model, pruned_module_info, config)
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
            for minfo in self.pruned_module_info:
                pruning_module = minfo.operand
                # 1. Calculate importance for all filters in all weights
                # 2. Calculate thresholds for every weight
                # 3. Set binary masks for filter
                filters_importance = self.filter_importance(minfo.module.weight)
                num_of_sparse_elems = get_rounded_pruned_element_number(filters_importance.size(0),
                                                                        self.pruning_rate)
                threshold = sorted(filters_importance)[num_of_sparse_elems]
                pruning_module.binary_filter_pruning_mask = calculate_binary_mask(filters_importance, threshold)

    def _set_binary_masks_for_all_pruned_modules(self):
        nncf_logger.debug("Setting new binary masks for all pruned modules together.")

        normalized_weights = []
        filter_importances = []
        for minfo in self.pruned_module_info:
            pruning_module = minfo.operand
            # 1. Calculate importance for all filters in all weights
            # 2. Calculate thresholds for every weight
            # 3. Set binary masks for filter
            normalized_weight = self.weights_normalizer(minfo.module.weight)
            normalized_weights.append(normalized_weight)

            filter_importances.append(self.filter_importance(normalized_weight))
        importances = torch.cat(filter_importances)
        threshold = sorted(importances)[int(self.pruning_rate * importances.size(0))]

        for i, minfo in enumerate(self.pruned_module_info):
            pruning_module = minfo.operand
            pruning_module.binary_filter_pruning_mask = calculate_binary_mask(filter_importances[i], threshold)

    def _apply_masks(self):
        nncf_logger.debug("Applying pruning binary masks")

        def _apply_binary_mask_to_module_weight_and_bias(module, mask, module_name=""):
            with torch.no_grad():
                # Applying mask to weights
                inplace_apply_filter_binary_mask(mask, module.weight, module_name)
                # Applying mask to bias too (if exists)
                if module.bias is not None:
                    inplace_apply_filter_binary_mask(mask, module.bias, module_name)

        for minfo in self.pruned_module_info:
            _apply_binary_mask_to_module_weight_and_bias(minfo.module, minfo.operand.binary_filter_pruning_mask,
                                                         minfo.module_name)

            # Applying mask to the BatchNorm node
            related_modules = minfo.related_modules
            if minfo.related_modules is not None and PrunedModuleInfo.BN_MODULE_NAME in minfo.related_modules \
                    and related_modules[PrunedModuleInfo.BN_MODULE_NAME] is not None:
                bn_module = related_modules[PrunedModuleInfo.BN_MODULE_NAME]
                _apply_binary_mask_to_module_weight_and_bias(bn_module, minfo.operand.binary_filter_pruning_mask)

            if minfo.related_modules is not None and PrunedModuleInfo.DEPTHWISE_BN_NAME in minfo.related_modules \
                    and related_modules[PrunedModuleInfo.DEPTHWISE_BN_NAME] is not None:
                bn_module = related_modules[PrunedModuleInfo.DEPTHWISE_BN_NAME]
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
