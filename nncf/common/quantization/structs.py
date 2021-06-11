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

from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional

from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName


class QuantizationMode:
    SYMMETRIC = 'symmetric'
    ASYMMETRIC = 'asymmetric'


class QuantizerConfig:
    """
    A generic, framework-agnostic information on a configuration of a quantizer for abstract reasoning
    and determination of a quantizer setup scheme for a given model.
    """

    def __init__(self, num_bits: int = 8,
                 mode: QuantizationMode = QuantizationMode.SYMMETRIC,
                 signedness_to_force: Optional[bool] = None,
                 per_channel: bool = False):
        """
        :param num_bits: Bitwidth of the quantization.
        :param mode: The mode of quantization (symmetric or asymmetric).
        :param signedness_to_force: True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param per_channel: True for per-channel quantization, False for per-tensor.
        """
        self.num_bits = num_bits
        self.mode = mode
        self.signedness_to_force = signedness_to_force
        self.per_channel = per_channel

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return 'B:{bits} M:{mode} SGN:{signedness} PC:{per_channel}'.format(
            bits=self.num_bits,
            mode='S' if self.mode == QuantizationMode.SYMMETRIC else 'A',
            signedness='ANY' if self.signedness_to_force is None else ('S' if self.signedness_to_force else 'U'),
            per_channel='Y' if self.per_channel else 'N')

    def __hash__(self):
        return hash(str(self))

    def is_valid_requantization_for(self, other: 'QuantizerConfig') -> bool:
        """
        Quantizer config A is a valid requantization for quantizer config B if A is more strict -
        specifically, it might be reasonable to put quantizer A after quantizer B in tensor data control flow, so that
        the requantization will further constrain the input tensor data w.r.t. values it can take, but
        putting quantizer A after quantizer B would be unreasonable.

        :param other: The "primary" QuantizerConfig, i.e. the one that defines an already present quantization.
        :return: True if the current config is a valid requantization for `other`, False otherwise.
        """
        fail_conditions = [
            self.num_bits > other.num_bits,
            self.mode is QuantizationMode.ASYMMETRIC and other.mode is QuantizationMode.SYMMETRIC,
            self.signedness_to_force is None and other.signedness_to_force is not None,
            self.signedness_to_force is True and other.signedness_to_force is False,
        ]
        if any(fail_conditions):
            return False
        return True

    def compatible_with_a_unified_scale_linked_qconfig(self, linked_qconfig: 'QuantizerConfig'):
        """
        For two configs to be compatible in a unified scale scenario, all of their fundamental parameters
        must be aligned.

        :param linked_qconfig: A QuantizerConfig that is compared against the current config.
        :return: A boolean value specifying whether `linked_qconfig` is compatible with the current config in terms
            of scale unification.
        """
        return self.num_bits == linked_qconfig.num_bits and \
               self.mode == linked_qconfig.mode and \
               self.signedness_to_force == linked_qconfig.signedness_to_force and \
               self.per_channel == linked_qconfig.per_channel

    def is_a_bitwidth_variant(self, other_qconfig: 'QuantizerConfig') -> bool:
        """
        :param other_qconfig: A QuantizerConfig to be compared against the current config.
        :return: A boolean value specifying whether `other_config` is identical to the current config
            in everything except the bitwidth.
        """
        return self.per_channel == other_qconfig.per_channel and \
               self.signedness_to_force == other_qconfig.signedness_to_force and \
               self.mode == other_qconfig.mode


class QuantizerSpec:
    """
    A specific (potentially framework-aware) parameter struct required to initialize a
    given object that performs quantization of an input tensor.
    """

    def __init__(self, num_bits: int,
                 mode: QuantizationMode,
                 signedness_to_force: bool,
                 narrow_range: bool,
                 half_range: bool):
        """
        :param num_bits: Bitwidth of the quantization.
        :param mode: The mode of quantization (symmetric or asymmetric).
        :param signedness_to_force: True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param narrow_range: True if the range of quantized values should be narrowed as compared to the
            naive case, False if all 2^`num_bits` quantizations should be used.
        :param half_range: If ``True`` effectively only a half of an quantizer range are used.
            False - the full range are used.
        """
        self.num_bits = num_bits
        self.mode = mode
        self.signedness_to_force = signedness_to_force
        self.narrow_range = narrow_range
        self.half_range = half_range

    def __eq__(self, other: 'QuantizerSpec'):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> 'QuantizerSpec':
        return cls(qconfig.num_bits,
                   qconfig.mode,
                   qconfig.signedness_to_force,
                   narrow_range,
                   half_range)


class QuantizationConstraints:
    REF_QCONF_OBJ = QuantizerConfig()

    def __init__(self, **kwargs):
        """
        Use attribute names of QuantizerConfig as arguments
        to set up constraints.
        E.g. QuantizationConstraint(bits=8, per_channel=True) will set up
        a constraint that corresponds to all 8-bit per-channel quantizers, either
        symmetric or asymmetric, either signed or unsigned.
        """
        for attr_name in kwargs:
            if not hasattr(QuantizationConstraints.REF_QCONF_OBJ, attr_name):
                raise RuntimeError('Invalid constraint - QuantizerConfig has no attribute \'{}\''.format(attr_name))
        self.qconf_attr_vs_constraint_dict = kwargs

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        is_compatible = True
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
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
    def from_config_dict(cls, config_dict: Dict) -> 'QuantizationConstraints':
        return cls(num_bits=config_dict.get('bits'),
                   mode=config_dict.get('mode'),
                   per_channel=config_dict.get('per_channel'),
                   signedness_to_force=config_dict.get('signed'))

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
    ACTIVATIONS = 'activations'
    WEIGHTS = 'weights'

    @staticmethod
    def from_str(str_: str) -> 'QuantizerGroup':
        if str_ == QuantizerGroup.ACTIVATIONS.value:
            return QuantizerGroup.ACTIVATIONS
        if str_ == QuantizerGroup.WEIGHTS.value:
            return QuantizerGroup.WEIGHTS
        raise RuntimeError('Unknown quantizer group string')


class QuantizableWeightedLayerNode:
    def __init__(self, node: NNCFNode, qconfig_list: List[QuantizerConfig]):
        self.node = node
        self.qconfig_list = qconfig_list


class QuantizerId:
    """
    Unique identifier of a quantizer. It's used to store and search all quantizers in a single
    structure.
    """

    def get_base(self):
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return str(self.get_base()) + self.get_suffix()

    def __hash__(self):
        return hash((self.get_base(), self.get_suffix()))

    def __eq__(self, other: 'QuantizerId'):
        return (self.get_base() == other.get_base()) and (self.get_suffix() == other.get_suffix())


class WeightQuantizerId(QuantizerId):
    """ Unique identifier of a quantizer for weights."""

    def __init__(self, target_node_name: NNCFNodeName):
        self.target_node_name = target_node_name

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return '|WEIGHT'


class NonWeightQuantizerId(QuantizerId):
    """
    Unique identifier of a quantizer, which corresponds to non-weight operations, such as
    ordinary activation, function and input
    """

    def __init__(self, target_node_name: NNCFNodeName,
                 input_port_id=None):
        self.target_node_name = target_node_name
        self.input_port_id = input_port_id

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return '|OUTPUT' if self.input_port_id is None else '|INPUT{}'.format(self.input_port_id)
