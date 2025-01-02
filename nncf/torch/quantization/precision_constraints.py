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
from typing import Dict, List

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerId


class HardwareQuantizationConstraints:
    def __init__(self):
        self._constraints: Dict[QuantizerId, List[QuantizerConfig]] = {}

    def add(self, quantizer_id: QuantizerId, qconfigs: List[QuantizerConfig]):
        self._constraints[quantizer_id] = qconfigs

    def get(self, quantizer_id: QuantizerId) -> List[QuantizerConfig]:
        if quantizer_id in self._constraints:
            return deepcopy(self._constraints[quantizer_id])
        return []

    def get_bitwidth_vs_qconfigs_dict(self, quantizer_id: QuantizerId) -> Dict[int, List[QuantizerConfig]]:
        bitwidths_vs_qconfigs: Dict[int, List[QuantizerConfig]] = {}
        for qc in self.get(quantizer_id):
            if qc.num_bits not in bitwidths_vs_qconfigs:
                bitwidths_vs_qconfigs[qc.num_bits] = [qc]
            else:
                bitwidths_vs_qconfigs[qc.num_bits].append(qc)
        return bitwidths_vs_qconfigs

    def replace(self, quantizer_id: QuantizerId, qconfig_set: List[QuantizerConfig]):
        if quantizer_id in self._constraints:
            self._constraints[quantizer_id] = qconfig_set

    def get_all_unique_bitwidths(self, qid: QuantizerId = None) -> List[int]:
        result = set()
        if qid is None:
            for qconfig_set in self._constraints.values():
                for qconfig in qconfig_set:
                    result.add(qconfig.num_bits)
        else:
            qconfs = self.get(qid)
            for qconfig in qconfs:
                result.add(qconfig.num_bits)
        return list(result)

    def __bool__(self):
        return bool(self._constraints)
