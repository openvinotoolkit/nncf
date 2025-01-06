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
from typing import Dict, List, NamedTuple, Tuple

from nncf.common.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import QuantizerSetupBase
from nncf.common.quantization.structs import QuantizerId
from nncf.torch.quantization.layers import BaseQuantizer


class AdjacentQuantizers(NamedTuple):
    """
    Combines activation and weight quantizers so that each quantizer is in the same group as the operation that it is
    affecting. Each quantizer that does not affect any node (e.g. if it only affects other quantizers as a topmost
    quantizer in a requantization scenario) will be placed in a separate group.
    :param: activation_quantizers   list of pairs of activation quantizers with their ids
    :param: weight_quantizers   list of pairs of weight quantizers with their ids
    """

    activation_quantizers: List[Tuple[QuantizerId, BaseQuantizer]]
    weight_quantizers: List[Tuple[QuantizerId, BaseQuantizer]]


class GroupsOfAdjacentQuantizers:
    """
    Contains groups of adjacent quantizers
    :param: weight_qp_id_per_activation_qp_id  gives a single activation quantizer for a given weight quantizer
    that directly quantize a weightable module (e.g. conv or linear)
    """

    def __init__(self):
        self.weight_qp_id_per_activation_qp_id: Dict[QuantizationPointId, QuantizationPointId] = {}
        self._quantizer_per_group_id = {}
        self._groups_of_adjacent_quantizers: List[AdjacentQuantizers] = []

    def get_group_id_for_quantizer(self, quantizer_id: QuantizerId):
        return self._quantizer_per_group_id.get(quantizer_id, None)

    def get_adjacent_quantizers_by_group_id(self, group_id):
        return (
            self._groups_of_adjacent_quantizers[group_id].weight_quantizers
            + self._groups_of_adjacent_quantizers[group_id].activation_quantizers
        )

    def __iter__(self):
        return iter(self._groups_of_adjacent_quantizers)

    def __bool__(self):
        return bool(self._groups_of_adjacent_quantizers) and bool(self._quantizer_per_group_id)

    def __getitem__(self, group_id):
        return self._groups_of_adjacent_quantizers[group_id]

    def parse_from_quantizer_setup(
        self,
        all_quantizations: Dict[QuantizerId, BaseQuantizer],
        quantizer_setup: QuantizerSetupBase,
        quantization_point_id_vs_quantizer_id: Dict[QuantizationPointId, QuantizerId],
    ):
        for group_idx, group in quantizer_setup.shared_input_operation_set_groups.items():
            act_quant_tuples: List[Tuple[QuantizerId, BaseQuantizer]] = []
            wt_quant_tuples: List[Tuple[QuantizerId, BaseQuantizer]] = []

            quantized_node_per_activation_qp_id: Dict[NNCFNodeName, QuantizationPointId] = {}
            module_scope_per_weight_qp_id: Dict[NNCFNodeName, QuantizationPointId] = {}

            for qp_id in group:
                qp = quantizer_setup.quantization_points[qp_id]
                quant_id = quantization_point_id_vs_quantizer_id[qp_id]
                quantizer_module = all_quantizations[quant_id]
                resulting_tuple = (quant_id, quantizer_module)
                if qp.is_weight_quantization_point():
                    wt_quant_tuples.append(resulting_tuple)
                    weight_quantized_module_node_name = qp.target_point.target_node_name
                    module_scope_per_weight_qp_id[weight_quantized_module_node_name] = qp_id
                elif qp.is_activation_quantization_point():
                    act_quant_tuples.append(resulting_tuple)
                    quantized_node_names = qp.directly_quantized_operator_node_names
                    quantized_node_per_activation_qp_id.update({node_name: qp_id for node_name in quantized_node_names})
                self._quantizer_per_group_id[quant_id] = group_idx

            for weight_quantized_module_node_name, w_qp_id in module_scope_per_weight_qp_id.items():
                if weight_quantized_module_node_name not in quantized_node_per_activation_qp_id:
                    nncf_logger.debug(
                        f"Module {weight_quantized_module_node_name} has quantized weights and no quantized inputs!"
                    )
                    continue
                a_qp_id = quantized_node_per_activation_qp_id[weight_quantized_module_node_name]
                if w_qp_id in self.weight_qp_id_per_activation_qp_id:
                    nncf_logger.debug(
                        f"Multiple weight quantizers per activation quantizer "
                        f"for {weight_quantized_module_node_name}"
                    )
                    continue
                self.weight_qp_id_per_activation_qp_id[w_qp_id] = a_qp_id

            adj_quants = AdjacentQuantizers(act_quant_tuples, wt_quant_tuples)
            self._groups_of_adjacent_quantizers.append(adj_quants)
