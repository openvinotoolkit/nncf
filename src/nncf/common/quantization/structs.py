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
from enum import Enum
from typing import Any, Optional

import nncf
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.utils.api_marker import api
from nncf.config.schemata.defaults import QUANTIZATION_BITS
from nncf.config.schemata.defaults import QUANTIZATION_NARROW_RANGE
from nncf.config.schemata.defaults import QUANTIZATION_PER_CHANNEL
from nncf.parameters import StrEnum
from nncf.parameters import TargetDevice


@api()
class QuantizationScheme(StrEnum):
    """
    Enumeration for specifying quantization schemes.

    :param SYMMETRIC: Symmetric quantization where the range is defined by a single parameter - scale.
        This range can include both negative and positive values if signed, or only positive values if unsigned.
    :param ASYMMETRIC: Asymmetric quantization where the range is defined by two parameters - input_low and input_high,
        representing the lower and upper boundaries of the range, respectively.
    :param SYMMETRIC_LORA: Symmetric quantization with Low-Rank Adapters (LoRA), involving the sum of weights and
        the multiplication of low-rank adapters.
    :param SYMMETRIC_LORA_NLS: Symmetric quantization with Low-Rank Adapters (LoRA) and Neural Low-Rank Adapter Search
        (NLS), involving the sum of weights and the multiplication of low-rank adapters.
    :param ASYMMETRIC_LORA: Asymmetric quantization with Low-Rank Adapters (LoRA), involving the sum of weights and
        the multiplication of low-rank adapters.
    :param ASYMMETRIC_LORA_NLS: Asymmetric quantization with Low-Rank Adapters (LoRA) and Neural Low-Rank Adapter Search
        (NLS), involving the sum of weights and the multiplication of low-rank adapters.
    """

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    SYMMETRIC_LORA = "symmetric_lora"
    SYMMETRIC_LORA_NLS = "symmetric_lora_nls"
    ASYMMETRIC_LORA = "asymmetric_lora"
    ASYMMETRIC_LORA_NLS = "asymmetric_lora_nls"


class QuantizerConfig:
    """
    A generic, framework-agnostic information on a configuration of a quantizer for abstract reasoning
    and determination of a quantizer setup scheme for a given model.
    """

    def __init__(
        self,
        num_bits: int = QUANTIZATION_BITS,
        mode: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        signedness_to_force: Optional[bool] = None,
        per_channel: bool = QUANTIZATION_PER_CHANNEL,
        narrow_range: bool = QUANTIZATION_NARROW_RANGE,
    ):
        """
        :param num_bits: Bitwidth of the quantization.
        :param mode: The mode of quantization (symmetric or asymmetric).
        :param signedness_to_force: True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param per_channel: True for per-channel quantization, False for per-tensor.
        :param narrow_range: True if the range of quantized values should be narrowed as compared to the
            naive case, False if all 2^`num_bits` quantizations should be used.
        """
        self.num_bits = num_bits
        self.mode = mode
        self.signedness_to_force = signedness_to_force
        self.per_channel = per_channel
        self.narrow_range = narrow_range

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantizerConfig):
            return False
        return self.__dict__ == other.__dict__

    def __str__(self) -> str:
        return "B:{bits} M:{mode} SGN:{signedness} PC:{per_channel} NR:{narrow_range}".format(
            bits=self.num_bits,
            mode="S" if self.mode == QuantizationScheme.SYMMETRIC else "A",
            signedness="ANY" if self.signedness_to_force is None else ("S" if self.signedness_to_force else "U"),
            per_channel="Y" if self.per_channel else "N",
            narrow_range="Y" if self.narrow_range else "N",
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other: "QuantizerConfig") -> bool:
        # Prefer higher bitwidths, per-tensor, symmetrical
        if self.num_bits > other.num_bits:
            return True
        if self.num_bits < other.num_bits:
            return False
        if self.per_channel is False and other.per_channel is True:
            return True
        if self.per_channel is True and other.per_channel is False:
            return False
        if self.mode is QuantizationScheme.SYMMETRIC and other.mode is QuantizationScheme.ASYMMETRIC:
            return True
        if self.mode is QuantizationScheme.ASYMMETRIC and other.mode is QuantizationScheme.SYMMETRIC:
            return False
        return False

    def is_redundant_with_downstream_qconfig(self, downstream_qconfig: "QuantizerConfig") -> bool:
        """
        Returns True if the two quantizers placed immediately one after another are redundant and could be
        replaced with a single quantizer.

        :param downstream_qconfig: QuantizerConfig of a quantizer placed immediately after the quantizer
            with the current QuantizerConfig.
        :return: True if the two quantizers placed immediately one after another are redundant and could be
            replaced with a single quantizer.
        """
        is_redundant = downstream_qconfig.num_bits == self.num_bits

        # Avoid asymmetric quantization if a symmetrically quantized tensor arrived
        is_redundant = is_redundant and (
            (downstream_qconfig.mode == self.mode)
            or (downstream_qconfig.mode == QuantizationScheme.ASYMMETRIC and self.mode == QuantizationScheme.SYMMETRIC)
        )

        # Avoid per-channel quantization if a per-tensor-quantized tensor arrived
        is_redundant = is_redundant and (
            (downstream_qconfig.per_channel == self.per_channel)
            or (downstream_qconfig.per_channel is True and self.per_channel is False)
        )

        # Strictly prohibit merging of config with different narrow_range params
        is_redundant = is_redundant and (downstream_qconfig.narrow_range == self.narrow_range)
        return is_redundant

    def is_compatible_with(self, other: "QuantizerConfig") -> bool:
        """
        Return True if the current QuantizerConfig and the other QuantizerConfig
        do not contradict each other.

        :param other: The QuantizerConfig to compare with.
        :return: True if the current QuantizerConfig and the other QuantizerConfig
            do not contradict each other.
        """
        return (
            self.per_channel == other.per_channel
            and self.mode == other.mode
            and self.num_bits == other.num_bits
            and self.narrow_range == other.narrow_range
            and (
                self.signedness_to_force == other.signedness_to_force
                or other.signedness_to_force is None
                or self.signedness_to_force is None
            )
        )

    def is_valid_requantization_for(self, other: "QuantizerConfig") -> bool:
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
            self.mode is QuantizationScheme.ASYMMETRIC and other.mode is QuantizationScheme.SYMMETRIC,
            self.signedness_to_force is None and other.signedness_to_force is not None,
            self.signedness_to_force is True and other.signedness_to_force is False,
            self.narrow_range != other.narrow_range,
        ]
        return not any(fail_conditions)

    def is_a_bitwidth_variant(self, other_qconfig: "QuantizerConfig") -> bool:
        """
        :param other_qconfig: A QuantizerConfig to be compared against the current config.
        :return: A boolean value specifying whether `other_config` is identical to the current config
            in everything except the bitwidth.
        """
        return (
            self.per_channel == other_qconfig.per_channel
            and self.signedness_to_force == other_qconfig.signedness_to_force
            and self.mode == other_qconfig.mode
        )

    def get_state(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            "num_bits": self.num_bits,
            "mode": self.mode,
            "signedness_to_force": self.signedness_to_force,
            "per_channel": self.per_channel,
            "narrow_range": self.narrow_range,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "QuantizerConfig":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class QuantizerSpec:
    """
    A specific (potentially framework-aware) parameter struct required to initialize a
    given object that performs quantization of an input tensor.
    """

    def __init__(
        self,
        num_bits: int,
        mode: QuantizationScheme,
        signedness_to_force: Optional[bool],
        narrow_range: Optional[bool],
        half_range: bool,
    ):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantizerSpec):
            return False
        return self.__dict__ == other.__dict__

    @classmethod
    def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> "QuantizerSpec":
        return cls(qconfig.num_bits, qconfig.mode, qconfig.signedness_to_force, narrow_range, half_range)


class QuantizationConstraints:
    REF_QCONF_OBJ = QuantizerConfig()

    def __init__(self, **kwargs: Any) -> None:
        """
        Use attribute names of QuantizerConfig as arguments
        to set up constraints.
        E.g. QuantizationConstraint(bits=8, per_channel=True) will set up
        a constraint that corresponds to all 8-bit per-channel quantizers, either
        symmetric or asymmetric, either signed or unsigned.
        """
        for attr_name in kwargs:
            if not hasattr(QuantizationConstraints.REF_QCONF_OBJ, attr_name):
                msg = f"Invalid constraint - QuantizerConfig has no attribute '{attr_name}'"
                raise nncf.ValidationError(msg)
        self.qconf_attr_vs_constraint_dict = kwargs

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                qconf_attr_value = getattr(qconfig, attr_name)
                if qconf_attr_value != constraint:
                    return False
        return True

    def get_updated_constraints(self, overriding_constraints: "QuantizationConstraints") -> "QuantizationConstraints":
        new_dict = deepcopy(self.qconf_attr_vs_constraint_dict)
        new_dict.update(overriding_constraints.qconf_attr_vs_constraint_dict)
        return QuantizationConstraints(**new_dict)

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any]) -> "QuantizationConstraints":
        return cls(
            num_bits=config_dict.get("bits"),
            mode=config_dict.get("mode"),
            per_channel=config_dict.get("per_channel"),
            signedness_to_force=config_dict.get("signed"),
        )

    def constrain_qconfig_list(
        self,
        node_name: NNCFNodeName,
        target_device: Optional[TargetDevice],
        quantizer_config_list: list[QuantizerConfig],
    ) -> list[QuantizerConfig]:
        assert quantizer_config_list is not None

        constrained_quantizer_config_list = list(filter(self.is_config_compatible, quantizer_config_list))

        # TODO: Make the logic more flexible when the flag "warning as error" is implemented.
        # It means that the qconfig from overrides must be selected as final config
        # even if it is not valid in hw-config.
        if not constrained_quantizer_config_list:
            err_msg = f"Quantization parameter constraints specified in NNCF config are incompatible \
            with HW capabilities as specified in HW config type '{target_device}'. \
            First conflicting quantizer location: {node_name}"
            raise ValueError(err_msg)

        return constrained_quantizer_config_list


class QuantizerGroup(Enum):
    ACTIVATIONS = "activations"
    WEIGHTS = "weights"


class QuantizableWeightedLayerNode:
    def __init__(self, node: NNCFNode, qconfig_list: list[QuantizerConfig]):
        self.node = node
        self.qconfig_list = qconfig_list


class QuantizerId:
    """
    Unique identifier of a quantizer. It's used to store and search all quantizers in a single
    structure.
    """

    def get_base(self) -> str:
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return str(self.get_base()) + self.get_suffix()

    def __hash__(self) -> int:
        return hash((self.get_base(), self.get_suffix()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantizerId):
            return False
        return (self.get_base() == other.get_base()) and (self.get_suffix() == other.get_suffix())


class WeightQuantizerId(QuantizerId):
    """Unique identifier of a quantizer for weights."""

    def __init__(self, target_node_name: NNCFNodeName):
        self.target_node_name = target_node_name

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return "|WEIGHT"


class NonWeightQuantizerId(QuantizerId):
    """
    Unique identifier of a quantizer, which corresponds to non-weight operations, such as
    ordinary activation, function and input
    """

    def __init__(self, target_node_name: NNCFNodeName, input_port_id: Optional[int] = None):
        self.target_node_name = target_node_name
        self.input_port_id = input_port_id

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return "|OUTPUT" if self.input_port_id is None else f"|INPUT{self.input_port_id}"


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


@api(canonical_alias="nncf.QuantizationPreset")
class QuantizationPreset(StrEnum):
    """
    An enum with values corresponding to the available quantization presets.
    """

    PERFORMANCE = "performance"
    MIXED = "mixed"

    def get_params_configured_by_preset(self, quant_group: QuantizerGroup) -> dict[str, str]:
        if quant_group == QuantizerGroup.ACTIVATIONS and self == QuantizationPreset.MIXED:
            return {"mode": QuantizationScheme.ASYMMETRIC}
        return {"mode": QuantizationScheme.SYMMETRIC}
