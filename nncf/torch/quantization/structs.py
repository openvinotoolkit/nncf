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
from typing import List

import torch

from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.quantization.layers import BaseQuantizer


class QuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer, affected_insertions: List[PTTargetPoint]):
        self.quantizer_module_ref = quantizer_module_ref
        self.affected_insertions = affected_insertions


class NonWeightQuantizerInfo(QuantizerInfo):
    pass


class WeightQuantizerInfo(QuantizerInfo):
    def __init__(
        self,
        quantizer_module_ref: BaseQuantizer,
        quantized_module: torch.nn.Module,
        affected_insertions: List[PTTargetPoint],
    ):
        super().__init__(quantizer_module_ref, affected_insertions)
        self.quantized_module = quantized_module
