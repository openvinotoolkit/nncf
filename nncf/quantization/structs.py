from collections import namedtuple

import torch
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Set

from nncf.nncf_network import InsertionInfo, InsertionPoint
from nncf.quantization.layers import QuantizerConfig, BaseQuantizer


class QuantizerSetupType(Enum):
    PATTERN_BASED = "pattern_based"
    PROPAGATION_BASED = "propagation_based"

    @staticmethod
    def from_str(quantizer_setup_type: str) -> 'QuantizerSetupType':
        if quantizer_setup_type == QuantizerSetupType.PATTERN_BASED.value:
            return QuantizerSetupType.PATTERN_BASED
        if quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED.value:
            return QuantizerSetupType.PROPAGATION_BASED
        raise RuntimeError("Unknown quantizer setup type. Please select 'pattern_based' or 'propagation_based'.")


class QuantizationConstraints:
    REF_QCONF_OBJ = QuantizerConfig()

    def __init__(self, **kwargs):
        """Use attribute names of QuantizerConfig as arguments
        to set up constraints.
        E.g. QuantizationConstraint(bits=8, per_channel=True) will set up
        a constraint that corresponds to all 8-bit per-channel quantizers, either
        symmetric or asymmetric, either signed or unsigned."""

        for attr_name in kwargs:
            if not hasattr(QuantizationConstraints.REF_QCONF_OBJ, attr_name):
                raise RuntimeError("Invalid constraint - QuantizerConfig has no attribute '{}'".format(attr_name))
        self.qconf_attr_vs_constraint_dict = kwargs

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        is_compatible = True
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if attr_name == 'logarithm_scale':
                continue  # Scale storage type is internal and should not affect HW config matching
            if constraint is not None:
                qconf_attr_value = getattr(qconfig, attr_name)
                if qconf_attr_value != constraint:
                    is_compatible = False
        return is_compatible

    def get_updated_constraints(self, overriding_constraints: 'QuantizationConstraints') -> 'QuantizationConstraints':
        new_dict = deepcopy(self.qconf_attr_vs_constraint_dict)
        new_dict.update(overriding_constraints.qconf_attr_vs_constraint_dict)
        return QuantizationConstraints(**new_dict)

    @classmethod
    def from_config_dict(cls, config_dict: Dict):
        return cls(bits=config_dict.get("bits"),
                   mode=config_dict.get("mode"),
                   per_channel=config_dict.get("per_channel"),
                   signedness_to_force=config_dict.get("signed"),
                   logarithm_scale=config_dict.get("logarithm_scale"))

    def constrain_qconfig_list(self, quantizer_config_list: List[QuantizerConfig]) -> List[QuantizerConfig]:
        assert quantizer_config_list is not None

        constrained_quantizer_config_list = list(filter(
            self.is_config_compatible,
            quantizer_config_list
        ))

        # TODO: Make the logic more flexible when the flag "warning as error" is implemented.
        # It means that the qconfig from overrides must be selected as final config
        # even if it is not valid in hw-config.
        if not constrained_quantizer_config_list:
            raise RuntimeError()

        return constrained_quantizer_config_list


class QuantizerGroup(Enum):
    ACTIVATIONS = "activations"
    WEIGHTS = "weights"

    @staticmethod
    def from_str(str_: str) -> 'QuantizerGroup':
        if str_ == QuantizerGroup.ACTIVATIONS.value:
            return QuantizerGroup.ACTIVATIONS
        if str_ == QuantizerGroup.WEIGHTS.value:
            return QuantizerGroup.WEIGHTS
        raise RuntimeError("Unknown quantizer group string")


QuantizableModule = namedtuple('QuantizableModule', 'module module_scope qconfig_list')


class NonWeightQuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer,
                 affected_insertions: List[InsertionInfo]):
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
