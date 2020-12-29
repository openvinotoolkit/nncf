"""
 Copyright (c) 2019-2020 Intel Corporation
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
from nncf.compression_method_api import CompressionAlgorithmController, CompressionLevel, StubCompressionScheduler
from nncf.nncf_network import NNCFNetwork
from nncf.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.sparsity.layers import BinaryMask
from nncf.sparsity.magnitude.functions import WEIGHT_IMPORTANCE_FUNCTIONS, calc_magnitude_binary_mask
from nncf.sparsity.schedulers import SPARSITY_SCHEDULERS


@COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, module):
        return BinaryMask(module.weight.size())

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        params = self.config.get("params", {})
        return MagnitudeSparsityController(target_model, self._sparsified_module_info,
                                           self.config,
                                           params.get('weight_importance', 'normed_abs'))


class MagnitudeSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork,
                 sparsified_module_info: List[SparseModuleInfo],
                 config, weight_importance: str):
        super().__init__(target_model, sparsified_module_info)
        self.config = config
        params = self.config.get("params", {})
        self.weight_importance = WEIGHT_IMPORTANCE_FUNCTIONS.get(weight_importance)
        self.sparsity_level_mode = params.get("sparsity_level_setting_mode", "global")
        self._scheduler = None
        self.sparsity_init = self.config.get("sparsity_init", 0)
        if self.sparsity_level_mode == 'global':
            scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
            self._scheduler = scheduler_cls(self, params)
        else:
            self._scheduler = StubCompressionScheduler()

        self.set_sparsity_level(self.sparsity_init)

    def statistics(self, quickly_collected_only=False):
        stats = super().statistics()
        if self.sparsity_level_mode == 'global':
            stats['sparsity_threshold'] =\
                 self._select_threshold(self.sparsity_rate_for_sparsified_modules, self.sparsified_module_info)
        else:
            table = Texttable()
            header = ["Name", "Per-layer sparsity threshold"]
            data = [header]

            for minfo in self.sparsified_module_info:
                drow = {h: 0 for h in header}
                drow["Name"] = minfo.module_name
                drow['Per-layer sparsity threshold'] =\
                     self._select_threshold(self.sparsity_rate_for_sparsified_modules, self.sparsified_module_info)
                row = [drow[h] for h in header]
                data.append(row)
            table.add_rows(data)
            stats['sparsity_thresholds'] = table
        return stats

    def freeze(self):
        for layer in self.sparsified_module_info:
            layer.operand.frozen = True

    def set_sparsity_level(self, sparsity_level,
                           target_sparsified_module_info: SparseModuleInfo = None,
                           run_batchnorm_adaptation: bool = False):
        if sparsity_level >= 1 or sparsity_level < 0:
            raise AttributeError(
                'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))
        if target_sparsified_module_info is None:
            target_sparsified_module_info_list = self.sparsified_module_info # List[SparseModuleInfo]
        else:
            target_sparsified_module_info_list = [target_sparsified_module_info]
        threshold = self._select_threshold(sparsity_level, target_sparsified_module_info_list)
        self._set_masks_for_threshold(threshold, target_sparsified_module_info_list)
        if run_batchnorm_adaptation:
            self.run_batchnorm_adaptation(self.config)

    def _select_threshold(self, sparsity_level, target_sparsified_module_info_list):
        all_weights = self._collect_all_weights(target_sparsified_module_info_list)
        if not all_weights:
            return 0.0
        all_weights_tensor, _ = torch.cat(all_weights).sort()
        threshold = all_weights_tensor[int((all_weights_tensor.size(0) - 1) * sparsity_level)].item()
        return threshold

    def _set_masks_for_threshold(self, threshold_val, target_sparsified_module_info_list):
        for layer in target_sparsified_module_info_list:
            if not layer.operand.frozen:
                layer.operand.binary_mask = calc_magnitude_binary_mask(layer.module.weight,
                                                                       self.weight_importance,
                                                                       threshold_val)


    def _collect_all_weights(self, target_sparsified_module_info_list: List[SparseModuleInfo]):
        all_weights = []
        for minfo in target_sparsified_module_info_list:
            all_weights.append(self.weight_importance(minfo.module.weight).view(-1))
        return all_weights

    def create_weight_sparsifying_operation(self, module):
        return BinaryMask(module.weight.size())

    def compression_level(self) -> CompressionLevel:
        if self.scheduler is not None:
            return self.scheduler.compression_level()
        return CompressionLevel.NONE

    def get_sparsity_init(self):
        return self.sparsity_init
