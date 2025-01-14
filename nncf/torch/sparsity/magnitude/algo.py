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
from copy import deepcopy
from typing import List

import torch

from nncf import NNCFConfig
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.common.graph import NNCFNode
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.sparsity.statistics import LayerThreshold
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.config.extractors import extract_algo_specific_config
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.config.schemata.defaults import MAGNITUDE_SPARSITY_WEIGHT_IMPORTANCE
from nncf.config.schemata.defaults import SPARSITY_INIT
from nncf.config.schemata.defaults import SPARSITY_LEVEL_SETTING_MODE
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoController
from nncf.torch.sparsity.base_algo import SparseModuleInfo
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.magnitude.functions import WEIGHT_IMPORTANCE_FUNCTIONS
from nncf.torch.sparsity.magnitude.functions import calc_magnitude_binary_mask


@PT_COMPRESSION_ALGORITHMS.register("magnitude_sparsity")
class MagnitudeSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        return BinaryMask(target_module_node.layer_attributes.get_weight_shape())

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MagnitudeSparsityController(model, self._sparsified_module_info, self.config)


@api()
@ADAPTIVE_COMPRESSION_CONTROLLERS.register("pt_magnitude_sparsity")
class MagnitudeSparsityController(BaseSparsityAlgoController):
    """
    Controller for the magnitude sparsity algorithm in PT.
    """

    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo], config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        self._config = config
        self._algo_config = extract_algo_specific_config(self._config, "magnitude_sparsity")
        params = self._algo_config.get("params", {})

        self._weight_importance_fn = WEIGHT_IMPORTANCE_FUNCTIONS[
            params.get("weight_importance", MAGNITUDE_SPARSITY_WEIGHT_IMPORTANCE)
        ]
        self._mode = params.get("sparsity_level_setting_mode", SPARSITY_LEVEL_SETTING_MODE)
        self._scheduler = None
        sparsity_init = self._algo_config.get("sparsity_init", SPARSITY_INIT)

        if self._mode == "global":
            scheduler_params = deepcopy(params)
            scheduler_params["sparsity_init"] = sparsity_init
            scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
            self._scheduler = scheduler_cls(self, scheduler_params)
        else:
            self._scheduler = StubCompressionScheduler()

        self._bn_adaptation = None

        self.set_sparsity_level(sparsity_init)

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()

        threshold_statistics = []
        if self._mode == "global":
            global_threshold = self._select_threshold(
                model_statistics.sparsity_level_for_layers, self.sparsified_module_info
            )

        module_name_to_sparsity_level_map = {
            s.name: s.sparsity_level for s in model_statistics.sparsified_layers_summary
        }
        for minfo in self.sparsified_module_info:
            if self._mode == "global":
                threshold = global_threshold
            else:
                sparsity_level_for_sparse_module = module_name_to_sparsity_level_map[minfo.module_node_name]
                threshold = self._select_threshold(sparsity_level_for_sparse_module, [minfo])

            threshold_statistics.append(LayerThreshold(minfo.module_node_name, threshold))

        target_sparsity_level = self.scheduler.current_sparsity_level if self._mode == "global" else None

        stats = MagnitudeSparsityStatistics(model_statistics, threshold_statistics, target_sparsity_level)

        nncf_stats = NNCFStatistics()
        nncf_stats.register("magnitude_sparsity", stats)
        return nncf_stats

    def freeze(self, freeze: bool = True):
        for layer in self.sparsified_module_info:
            layer.operand.frozen = freeze

    @property
    def compression_rate(self):
        return self.statistics().magnitude_sparsity.model_statistics.sparsity_level

    @compression_rate.setter
    def compression_rate(self, sparsity_level: float):
        self.freeze(False)
        self.set_sparsity_level(sparsity_level)
        self.freeze(True)

    def set_sparsity_level(
        self,
        sparsity_level,
        target_sparsified_module_info: SparseModuleInfo = None,
        run_batchnorm_adaptation: bool = False,
    ):
        if sparsity_level >= 1 or sparsity_level < 0:
            raise AttributeError(
                "Sparsity level should be within interval [0,1), actual value to set is: {}".format(sparsity_level)
            )
        if target_sparsified_module_info is None:
            target_sparsified_module_info_list = self.sparsified_module_info  # List[SparseModuleInfo]
        else:
            target_sparsified_module_info_list = [target_sparsified_module_info]
        threshold = self._select_threshold(sparsity_level, target_sparsified_module_info_list)
        self._set_masks_for_threshold(threshold, target_sparsified_module_info_list)

        if run_batchnorm_adaptation:
            self._run_batchnorm_adaptation()

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
                layer.operand.binary_mask = calc_magnitude_binary_mask(
                    layer.module.weight, self._weight_importance_fn, threshold_val
                )

    def _collect_all_weights(self, target_sparsified_module_info_list: List[SparseModuleInfo]):
        all_weights = []
        for minfo in target_sparsified_module_info_list:
            all_weights.append(self._weight_importance_fn(minfo.module.weight).view(-1))
        return all_weights

    def compression_stage(self) -> CompressionStage:
        if self._mode == "local":
            return CompressionStage.FULLY_COMPRESSED

        if self.scheduler.current_sparsity_level >= self.scheduler.target_level:
            return CompressionStage.FULLY_COMPRESSED
        if self.scheduler.current_sparsity_level == 0:
            return CompressionStage.UNCOMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self._config, "magnitude_sparsity")
            )
        self._bn_adaptation.run(self.model)
