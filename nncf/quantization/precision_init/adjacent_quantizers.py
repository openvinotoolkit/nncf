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
from typing import List, Tuple, NamedTuple, Dict

from nncf.quantization.layers import BaseQuantizer
from nncf.quantization.quantizer_id import QuantizerId
from nncf.quantization.structs import QuantizationPointId, QuantizerSetupBase


class AdjacentQuantizers(NamedTuple):
    activation_quantizers: List[Tuple[QuantizerId, BaseQuantizer]]
    weight_quantizers: List[Tuple[QuantizerId, BaseQuantizer]]


class GroupsOfAdjacentQuantizers:
    def __init__(self):
        self._quantizer_per_group_id = {}
        self._groups_of_adjacent_quantizers: List[AdjacentQuantizers] = []

    def get_group_id_for_quantizer(self, quantizer_id: QuantizerId):
        return self._quantizer_per_group_id.get(quantizer_id, None)

    def get_adjacent_quantizers_by_group_id(self, group_id):
        return self._groups_of_adjacent_quantizers[group_id].weight_quantizers + \
                self._groups_of_adjacent_quantizers[group_id].activation_quantizers

    def __iter__(self):
        return iter(self._groups_of_adjacent_quantizers)

    def __bool__(self):
        return bool(self._groups_of_adjacent_quantizers) and bool(self._quantizer_per_group_id)

    def __getitem__(self, group_id):
        return self._groups_of_adjacent_quantizers[group_id]

    def parse_from_quantizer_setup(self, all_quantizations: Dict[QuantizerId, BaseQuantizer],
                                   quantizer_setup: QuantizerSetupBase,
                                   quantization_point_id_vs_quantizer_id: Dict[QuantizationPointId, QuantizerId]):

        for group_idx, group in enumerate(quantizer_setup.shared_input_operation_set_groups):
            act_quant_tuples = []  # type: List[Tuple[QuantizerId, BaseQuantizer]]
            wt_quant_tuples = []  # type: List[Tuple[QuantizerId, BaseQuantizer]]
            for qp_id in group:
                quant_id = quantization_point_id_vs_quantizer_id[qp_id]
                quantizer_module = all_quantizations[quant_id]
                resulting_tuple = (quant_id, quantizer_module)
                if quantizer_setup.quantization_points[qp_id].is_weight_quantization_point():
                    wt_quant_tuples.append(resulting_tuple)
                elif quantizer_setup.quantization_points[qp_id].is_activation_quantization_point():
                    act_quant_tuples.append(resulting_tuple)
                self._quantizer_per_group_id[quant_id] = group_idx

            adj_quants = AdjacentQuantizers(act_quant_tuples, wt_quant_tuples)
            self._groups_of_adjacent_quantizers.append(adj_quants)
