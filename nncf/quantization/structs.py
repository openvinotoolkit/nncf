import torch
from typing import List, Set

from nncf.nncf_network import InsertionPoint
from nncf.quantization.layers import BaseQuantizer


class NonWeightQuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer,
                 affected_insertions: List[InsertionPoint]):
        self.quantizer_module_ref = quantizer_module_ref
        self.affected_insertions = affected_insertions


class WeightQuantizerInfo:
    def __init__(self,
                 quantizer_module_ref: BaseQuantizer,
                 quantized_module: torch.nn.Module):
        self.quantizer_module_ref = quantizer_module_ref
        self.quantized_module = quantized_module


class QuantizersBetweenQuantizableLayers:
    """ Contains locations of quantizers between inputs quantizable layers: input agnostic operation execution context
    for activations and scope - for quantized modules """

    def __init__(self):
        self.activation_quantizer_insertion_points = set()  # type: Set[InsertionPoint]
        self.quantized_module_scopes = set()  # type: Set['Scope']

    def add_activation_quantizer_insertion_point(self, ip: InsertionPoint):
        self.activation_quantizer_insertion_points.add(ip)

    def add_quantized_module_scope(self, scope: 'Scope'):
        self.quantized_module_scopes.add(scope)

    def __bool__(self) -> bool:
        return bool(self.activation_quantizer_insertion_points) and bool(self.quantized_module_scopes)

    def update(self, other: 'QuantizersBetweenQuantizableLayers'):
        self.activation_quantizer_insertion_points.update(other.activation_quantizer_insertion_points)
        self.quantized_module_scopes.update(other.quantized_module_scopes)
