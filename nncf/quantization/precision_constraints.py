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
from copy import deepcopy
from typing import Dict, List, Set

from .quantizer_id import QuantizerId


class PrecisionConstraints:
    def __init__(self):
        self._constraints = {}  # type: Dict[QuantizerId, Set[int]]

    def add(self, quantizer_id: QuantizerId, bits_set: Set[int]):
        self._constraints[quantizer_id] = bits_set

    def get(self, quantizer_id: QuantizerId) -> Set[int]:
        if quantizer_id in self._constraints:
            return deepcopy(self._constraints[quantizer_id])
        return set()

    def replace(self, quantizer_id: QuantizerId, bits: Set[int]):
        if quantizer_id in self._constraints:
            self._constraints[quantizer_id] = bits

    def get_all_unique_bits(self) -> List[int]:
        result = set()
        for bits_set in self._constraints.values():
            result.update(bits_set)
        return list(result)

    def __bool__(self):
        return bool(self._constraints)
