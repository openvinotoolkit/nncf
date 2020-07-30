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

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmController, CompressionLevel
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
        scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
        self._scheduler = scheduler_cls(self, params)

    def statistics(self):
        stats = super().statistics()
        stats['sparsity_threshold'] = self._select_threshold(self.sparsity_rate_for_sparsified_modules)
        return stats

    def freeze(self):
        pass

    def set_sparsity_level(self, sparsity_level):
        if sparsity_level >= 1 or sparsity_level < 0:
            raise AttributeError(
                'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))
        self._set_masks_for_threshold(self._select_threshold(sparsity_level))
        self.run_batchnorm_adaptation(self.config)

    def _select_threshold(self, sparsity_level):
        all_weights = self._collect_all_weights()
        if not all_weights:
            return 0.0
        all_weights_tensor, _ = torch.cat(all_weights).sort()
        threshold = all_weights_tensor[int((all_weights_tensor.size(0) - 1) * sparsity_level)].item()
        return threshold

    def _set_masks_for_threshold(self, threshold_val):
        for layer in self.sparsified_module_info:
            layer.operand.binary_mask = calc_magnitude_binary_mask(layer.module.weight,
                                                                   self.weight_importance,
                                                                   threshold_val)

    def _collect_all_weights(self):
        all_weights = []
        for minfo in self.sparsified_module_info:
            all_weights.append(self.weight_importance(minfo.module.weight).view(-1))
        return all_weights

    def create_weight_sparsifying_operation(self, module):
        return BinaryMask(module.weight.size())

    def compression_level(self) -> CompressionLevel:
        return self.scheduler.compression_level()
