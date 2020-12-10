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
from typing import List, Dict

from nncf.nncf_network import NNCFNetwork

from nncf.quantization.layers import QUANTIZATION_MODULES, BaseQuantizer
from nncf.quantization.precision_init.base_init import WeightQuantizersHandler
from nncf.quantization.precision_constraints import PrecisionConstraints
from nncf.quantization.quantizer_id import QuantizerId
from nncf.utils import get_all_modules_by_type


class CompressionRatioCalculator:
    """
    Calculates compression ratio - ratio between bits complexity of fully INT8 model and mixed-precision lower-bit one.
    Bit complexity of the model is a sum of bit complexities for each quantized layer, which are a multiplication of
    FLOPS for the layer by number of bits for its quantization. The compression ratio can be used for estimation of
    performance boost for quantized model.
    """
    DEFAULT_NUMBER_OF_BITS = 8

    def __init__(self, model: NNCFNetwork, quantizers_handler: WeightQuantizersHandler):
        flops_count_per_module_scope = model.get_flops_per_module()

        self._weight_quantizers_in_exec_order = quantizers_handler.get_weight_quantizers_in_execution_order_per_id()

        self.ops_per_quantizer_id = {}
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        all_quantizers_in_model = get_all_modules_by_type(model.get_nncf_wrapped_model(), quantization_types)

        for scope in all_quantizers_in_model:
            if quantizers_handler.is_wq_scope(scope):
                quantizer_id = quantizers_handler.get_quantizer_id_by_scope(scope)
                affected_module_scope = quantizers_handler.get_owning_module_scope_from_wq_scope(scope)
                self.ops_per_quantizer_id[quantizer_id] = flops_count_per_module_scope[affected_module_scope]

        self.total_ops_count = sum(v for v in self.ops_per_quantizer_id.values()) * self.DEFAULT_NUMBER_OF_BITS

    def ratio_for_bits_configuration(self, execution_order_bits_config: List[int],
                                     skipped: Dict[QuantizerId, BaseQuantizer] = None) -> float:
        """
        Calculates compression ratio for a given bits configuration

        Args:
            execution_order_bits_config: list of bits for each weight quantization in the order of execution
            skipped: quantizers that were skipped from bitwidth initialization, since their bitwidth is determined
            unambiguously based on constraints of the HW config

        Returns:
            compression ratio of mixed-precision model by relation to fully INT8
        """
        quantizer_ops = 0
        for num_bits, (quantizer_id, quantizer) in zip(execution_order_bits_config,
                                                       self._weight_quantizers_in_exec_order.items()):
            quantizer_ops += num_bits * self.ops_per_quantizer_id[quantizer_id]
        if skipped:
            for quantizer_id, quantizer in skipped.items():
                quantizer_ops += quantizer.num_bits * self.ops_per_quantizer_id[quantizer_id]

        return self.total_ops_count / quantizer_ops

    def ratio_limits(self, bits: List[int], constraints: PrecisionConstraints = None,
                     skipped: Dict[QuantizerId, BaseQuantizer] = None) -> (float, float):
        """
        Calculates minimum and maximum compression ratio.

        Args:
            bits: list of all available bits for weight quantization
            constraints: precision constraints defined by HW config
            skipped: quantizers that were skipped from bitwidth initialization, since their bitwidth is determined
            unambiguously based on constraints of the HW config

        Returns:
            minimum and maximum compression ratio
        """
        config_len = len(self._weight_quantizers_in_exec_order)
        min_config = [min(bits)] * config_len
        max_config = [max(bits)] * config_len
        if constraints:
            for i, quantizer_id in enumerate(self._weight_quantizers_in_exec_order):
                bit_constraints = constraints.get(quantizer_id)
                if bit_constraints:
                    min_config[i] = min(bit_constraints)
                    max_config[i] = max(bit_constraints)

        max_ratio = self.ratio_for_bits_configuration(min_config, skipped)
        min_ratio = self.ratio_for_bits_configuration(max_config, skipped)
        return min_ratio, max_ratio
