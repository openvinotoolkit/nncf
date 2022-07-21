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

from nncf.experimental.torch.fracbits.controller import FracBitsQuantizationController
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import QuantizationBuilder
from nncf.torch.quantization.layers import PTQuantizerSetup
from nncf.common.quantization.structs import QuantizationMode
from nncf.experimental.torch.fracbits.structs import FracBitsQuantizationMode


@PT_COMPRESSION_ALGORITHMS.register('fracbits_quantization')
class FracBitsQuantizationBuilder(QuantizationBuilder):
    def _get_quantizer_setup(self, target_model: NNCFNetwork) -> PTQuantizerSetup:
        setup = super()._get_quantizer_setup(target_model)

        for q_point in setup.quantization_points.values():
            mode = q_point.qspec.mode
            if mode == QuantizationMode.ASYMMETRIC:
                q_point.qspec.mode = FracBitsQuantizationMode.ASYMMETRIC
            elif mode == QuantizationMode.SYMMETRIC:
                q_point.qspec.mode = FracBitsQuantizationMode.SYMMETRIC
            else:
                raise ValueError(f"qsepc.mode={mode} is unknown.")

        return setup

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return FracBitsQuantizationController(model,
                                              self.config,
                                              self._debug_interface,
                                              self._weight_quantizers,
                                              self._non_weight_quantizers,
                                              self._groups_of_adjacent_quantizers,
                                              self._quantizers_input_shapes,
                                              build_time_metric_info=self._build_time_metric_infos,
                                              build_time_range_init_params=self._range_init_params)
