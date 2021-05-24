"""
 Copyright (c) 2021 Intel Corporation
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
from enum import Enum
from typing import List

import torch

from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.quantization.layers import BaseQuantizer


class QuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer,
                 affected_insertions: List[PTTargetPoint]):
        self.quantizer_module_ref = quantizer_module_ref
        self.affected_insertions = affected_insertions


class NonWeightQuantizerInfo(QuantizerInfo):
    pass


class WeightQuantizerInfo(QuantizerInfo):
    def __init__(self,
                 quantizer_module_ref: BaseQuantizer,
                 quantized_module: torch.nn.Module,
                 affected_insertions: List[PTTargetPoint]):
        super().__init__(quantizer_module_ref, affected_insertions)
        self.quantized_module = quantized_module


class UnifiedScaleType(Enum):
    """
    UNIFY_ONLY_PER_TENSOR - only results in scale unification if per-tensor quantization is ultimately applied.
    This is the target scenario for concat unified scales since the channel count between the concatenated tensors
    may be mismatching and, more importantly, the concatenation might occur on exactly the channel dimension which
    means that the concatenated tensor must reuse all quantization scales of the input per-channel
    quantized tensors.
    UNIFY_ALWAYS - results in scale unification for both per-channel and per-tensor quantization. This is the
    target scenario for eltwise unified scales, as it is assumed that the eltwise ops have matching input
    tensor shapes and therefore the quantization channel count is the same.
    """
    UNIFY_ONLY_PER_TENSOR = 0
    UNIFY_ALWAYS = 1
