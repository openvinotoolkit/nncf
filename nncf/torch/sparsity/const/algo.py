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
from typing import Tuple

from nncf.common.graph import NNCFNode
from nncf.common.sparsity.statistics import ConstSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoController
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.torch.sparsity.layers import BinaryMask


@PT_COMPRESSION_ALGORITHMS.register("const_sparsity")
class ConstSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        return BinaryMask(target_module_node.layer_attributes.get_weight_shape())

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return ConstSparsityController(model, self._sparsified_module_info)

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        return True, "Frozen layers are allowed for const sparsity"


@api()
class ConstSparsityController(BaseSparsityAlgoController):
    """
    Controller for the auxiliary constant sparsity algorithm in PT.
    """

    def freeze(self):
        pass

    def set_sparsity_level(self, sparsity_level: float):
        pass

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()
        stats = ConstSparsityStatistics(model_statistics)

        nncf_stats = NNCFStatistics()
        nncf_stats.register("const_sparsity", stats)
        return nncf_stats
