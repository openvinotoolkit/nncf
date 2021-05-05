from enum import Enum
from typing import List

import torch

from nncf.graph.transformations.commands import PTTargetPoint
from nncf.quantization.layers import BaseQuantizer


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
