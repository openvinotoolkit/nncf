from typing import List

import torch

from nncf.dynamic_graph.transformations.commands import PTTargetPoint
from nncf.quantization.layers import BaseQuantizer


class NonWeightQuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer,
                 affected_insertions: List[PTTargetPoint]):
        self.quantizer_module_ref = quantizer_module_ref
        self.affected_insertions = affected_insertions


class WeightQuantizerInfo:
    def __init__(self,
                 quantizer_module_ref: BaseQuantizer,
                 quantized_module: torch.nn.Module,
                 affected_insertions: List[PTTargetPoint]):
        self.quantizer_module_ref = quantizer_module_ref
        self.quantized_module = quantized_module
        self.affected_insertions = affected_insertions
