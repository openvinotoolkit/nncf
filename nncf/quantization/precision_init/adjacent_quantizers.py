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
from collections import OrderedDict
from typing import List, Tuple, NamedTuple

from nncf.quantization.layers import BaseQuantizer
from ..quantizer_id import QuantizerId
from ..quantizer_propagation import QuantizersBetweenQuantizableLayers


class AdjacentQuantizers(NamedTuple):
    activation_quantizers: List[Tuple[QuantizerId, BaseQuantizer]]
    weight_quantizers: List[Tuple[QuantizerId, BaseQuantizer]]


class GroupsOfAdjacentQuantizers:
    def __init__(self, quantization_ctrl: 'QuantizationController'):
        repeated_groups = []
        non_weight_quantizers = quantization_ctrl.non_weight_quantizers
        sorted_quantizers = OrderedDict(sorted(non_weight_quantizers.items(), key=lambda x: str(x[0])))
        for quantizer_id, quantizer_info in sorted_quantizers.items():
            group = quantizer_info.quantizers_between_quantizable_layers  # type: QuantizersBetweenQuantizableLayers
            if group:
                repeated_groups.append(group)

        self._quantizer_per_group_id = {}
        self._groups_of_adjacent_quantizers: List[AdjacentQuantizers] = []

        unique_groups = list(dict.fromkeys(repeated_groups))

        for i, group in enumerate(unique_groups):
            quantized_module_scopes = group.quantized_module_scopes
            paired_wq = []
            for scope in quantized_module_scopes:
                for quantizer_id, quantizer in quantization_ctrl.weight_quantizers.items():
                    if scope == quantizer_id.get_scope():
                        paired_wq.append((quantizer_id, quantizer))
                        self._quantizer_per_group_id[id(quantizer)] = i
                        break
            paired_aq = []
            for ia_op_ctx in group.activation_quantizer_ctxs:
                for quantizer_id, quantizer_info in quantization_ctrl.non_weight_quantizers.items():
                    if ia_op_ctx == quantizer_id.ia_op_exec_context:
                        quantizer = quantizer_info.quantizer_module_ref
                        paired_aq.append((quantizer_id, quantizer))
                        self._quantizer_per_group_id[id(quantizer)] = i
                        break

            self._groups_of_adjacent_quantizers.append(AdjacentQuantizers(paired_aq, paired_wq))

    def get_group_id_for_quantizer(self, quantizer: BaseQuantizer):
        qid = id(quantizer)
        return self._quantizer_per_group_id.get(qid, None)

    def __iter__(self):
        return iter(self._groups_of_adjacent_quantizers)

    def __bool__(self):
        return bool(self._groups_of_adjacent_quantizers) and bool(self._quantizer_per_group_id)
