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

from contextlib import contextmanager
from typing import Dict, Tuple

from nncf.common.quantization.structs import NonWeightQuantizerId, QuantizerId, WeightQuantizerId
from nncf.common.statistics import NNCFStatistics
from nncf.config.config import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.experimental.torch.fracbits.statistics import FracBitsStatistics
from nncf.experimental.torch.fracbits.scheduler import FracBitsQuantizationScheduler
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import QuantizationController, QuantizationDebugInterface
from nncf.torch.quantization.init_range import PTRangeInitParams
from nncf.torch.quantization.metrics import QuantizationShareBuildTimeInfo
from nncf.torch.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.torch.quantization.structs import NonWeightQuantizerInfo, WeightQuantizerInfo
from nncf.experimental.torch.fracbits.loss import FRACBITS_LOSSES


class FracBitsQuantizationController(QuantizationController):
    def __init__(self, target_model: NNCFNetwork,
                 config: NNCFConfig,
                 debug_interface: QuantizationDebugInterface,
                 weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
                 groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                 quantizers_input_shapes: Dict[QuantizerId, Tuple[int]],
                 build_time_metric_info: QuantizationShareBuildTimeInfo = None,
                 build_time_range_init_params: PTRangeInitParams = None):
        super().__init__(target_model, config, debug_interface, weight_quantizers, non_weight_quantizers,
                         groups_of_adjacent_quantizers, quantizers_input_shapes,
                         build_time_metric_info, build_time_range_init_params)
        self._set_fracbits_loss(target_model)
        self._set_scheduler()

    def _set_fracbits_loss(self, target_model: NNCFNetwork):
        algo_config = self._get_algo_config()

        if algo_config.get("loss") is None:
            raise RuntimeError("You didn't set loss config.")

        loss_config = algo_config.get("loss")

        if loss_config.get("type") is None:
            raise RuntimeError("You didn't set loss.type config.")

        if loss_config.get("compression_rate") is None:
            raise RuntimeError("You didn't set compression_rate config.")

        if loss_config.get("criteria") is None:
            raise RuntimeError("You didn't set criteria config.")

        loss_type = loss_config.get("type")
        compression_rate = loss_config.get("compression_rate")
        criteria = loss_config.get("criteria")
        flip_loss = loss_config.get("flip_loss")
        alpha = loss_config.get("alpha")

        self._loss: PTCompressionLoss = FRACBITS_LOSSES.get(loss_type)(
            model=target_model, compression_rate=compression_rate, criteria=criteria, flip_loss=flip_loss, alpha=alpha)

    def _set_scheduler(self):
        algo_config = self._get_algo_config()

        freeze_epoch = algo_config.get("freeze_epoch", None)

        if freeze_epoch is None:
            raise RuntimeError("You didn't set freeze_epoch config.")

        def _callback():
            self.freeze_bit_widths()

        self._scheduler = FracBitsQuantizationScheduler(
            freeze_callback=_callback,
            freeze_epoch=freeze_epoch
        )

    def _get_algo_config(self) -> Dict:
        return extract_algo_specific_config(self.config, algo_name_to_match="fracbits_quantization")

    def freeze_bit_widths(self):
        for q in self.all_quantizations.values():
            q.freeze_num_bits()

        # We need to broadcast bit widths to synchronize if it is under distributed system.
        # If you remove this you can get the following error.
        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one...
        self._broadcast_initialized_params_for_each_quantizer()

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        @contextmanager
        def _base_name_context():
            tmp_name = self._name
            self._name = "quantization"
            yield self.name
            self._name = tmp_name

        with _base_name_context():
            nncf_statistics = super().statistics(quickly_collected_only)

        nncf_statistics.register(self.name, FracBitsStatistics(self._loss.get_state()))

        return nncf_statistics
