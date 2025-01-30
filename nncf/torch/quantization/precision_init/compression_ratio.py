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
from typing import Dict

from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup


class CompressionRatioCalculator:
    """
    Calculates compression ratio - ratio between bits complexity of fully INT8 model and mixed-precision lower-bit one.
    Bit complexity of the model is a sum of bit complexities for each quantized layer, which are a multiplication of
    FLOPS for the layer by number of bits for its quantization. The compression ratio can be used for estimation of
    performance boost for quantized model.
    """

    DEFAULT_NUMBER_OF_BITS = 8

    def __init__(
        self,
        flops_per_weighted_module_node: Dict[NNCFNodeName, int],
        quantizer_setup: SingleConfigQuantizerSetup,
        weight_qp_id_per_activation_qp_id: Dict[QuantizationPointId, QuantizationPointId],
    ):
        self._weight_qp_id_per_activation_qp_id = weight_qp_id_per_activation_qp_id
        self._flops_per_weight_qp_id: Dict[QuantizationPointId, float] = {}
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_weight_quantization_point():
                target_node_name = qp.insertion_point.target_node_name
                self._flops_per_weight_qp_id[qp_id] = flops_per_weighted_module_node[target_node_name]
        self.maximum_bits_complexity = sum(self._flops_per_weight_qp_id.values()) * self.DEFAULT_NUMBER_OF_BITS

    def run_for_quantizer_setup(self, quantizer_setup: SingleConfigQuantizerSetup) -> float:
        """
        Calculates compression ratio for a given quantizer setup with
        :param: quantizer_setup: setup with information quantization points
        :returns: compression ratio of mixed-precision model by relation to fully INT8
        """
        quantization_points = quantizer_setup.quantization_points
        weight_qps = list(filter(lambda pair: pair[1].is_weight_quantization_point(), quantization_points.items()))
        bits_complexity = 0
        for w_qp_id, w_qp in weight_qps:
            wq_num_bits = w_qp.qconfig.num_bits
            a_qp_id = self._weight_qp_id_per_activation_qp_id[w_qp_id]
            a_qp = quantization_points[a_qp_id]
            aq_num_bits = a_qp.qconfig.num_bits
            num_bits = max(wq_num_bits, aq_num_bits)
            bits_complexity += num_bits * self._flops_per_weight_qp_id[w_qp_id]
        return self.maximum_bits_complexity / bits_complexity
