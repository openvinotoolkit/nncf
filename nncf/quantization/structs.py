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
