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
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import jstyleson as json  # type: ignore[import-untyped]

import nncf
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging import nncf_logger
from nncf.common.quantization import quantizers as quant
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.helpers import product_dict
from nncf.common.utils.os import safe_open
from nncf.definitions import HW_CONFIG_RELATIVE_DIR
from nncf.definitions import NNCF_PACKAGE_ROOT_DIR


class HWConfigType(Enum):
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"


HW_CONFIG_TYPE_TARGET_DEVICE_MAP = {
    "ANY": HWConfigType.CPU.value,
    "CPU": HWConfigType.CPU.value,
    "NPU": HWConfigType.NPU.value,
    "GPU": HWConfigType.GPU.value,
    "CPU_SPR": HWConfigType.CPU.value,
}


HWConfigOpName = str


def get_hw_config_type(target_device: str) -> Optional[HWConfigType]:
    """
    Returns hardware configuration type for target device

    :param target_device: A target device
    :raises ValueError: if target device is not supported yet
    :return: hardware configuration type or None for the 'TRIAL' target device
    """
    if target_device == "TRIAL":
        return None
    return HWConfigType(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])


class HWConfig(list[Dict[str, Any]], ABC):
    QUANTIZATION_ALGORITHM_NAME = "quantization"
    ATTRIBUTES_NAME = "attributes"
    SCALE_ATTRIBUTE_NAME = "scales"
    UNIFIED_TYPE_NAME = "unified"
    ADJUST_PADDING_ATTRIBUTE_NAME = "adjust_padding"

    TYPE_TO_CONF_NAME_DICT = {HWConfigType.CPU: "cpu.json", HWConfigType.NPU: "npu.json", HWConfigType.GPU: "gpu.json"}

    def __init__(self) -> None:
        super().__init__()
        self.registered_algorithm_configs: Dict[str, Any] = {}
        self.target_device = None

    @abstractmethod
    def _get_available_operator_metatypes_for_matching(self) -> List[Type[OperatorMetatype]]:
        pass

    @staticmethod
    def get_path_to_hw_config(hw_config_type: HWConfigType) -> str:
        return "/".join(
            [NNCF_PACKAGE_ROOT_DIR, HW_CONFIG_RELATIVE_DIR, HWConfig.TYPE_TO_CONF_NAME_DICT[hw_config_type]]
        )

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "HWConfig":
        hw_config = cls()
        hw_config.target_device = dct["target_device"]

        for algorithm_name, algorithm_configs in dct.get("config", {}).items():
            hw_config.registered_algorithm_configs[algorithm_name] = {}
            for algo_config_alias, algo_config in algorithm_configs.items():
                for key, val in algo_config.items():
                    if not isinstance(val, list):
                        algo_config[key] = [val]

                hw_config.registered_algorithm_configs[algorithm_name][algo_config_alias] = list(
                    product_dict(algo_config)
                )

        for op_dict in dct.get("operations", []):
            for algorithm_name in op_dict:
                if algorithm_name not in hw_config.registered_algorithm_configs:
                    continue
                tmp_config: Dict[str, List[Dict[str, Any]]] = {}
                for algo_and_op_specific_field_name, algorithm_configs in op_dict[algorithm_name].items():
                    if not isinstance(algorithm_configs, list):
                        algorithm_configs = [algorithm_configs]

                    tmp_config[algo_and_op_specific_field_name] = []
                    for algorithm_config in algorithm_configs:
                        if isinstance(algorithm_config, str):  # Alias was supplied
                            tmp_config[algo_and_op_specific_field_name].extend(
                                hw_config.registered_algorithm_configs[algorithm_name][algorithm_config]
                            )
                        else:
                            for key, val in algorithm_config.items():
                                if not isinstance(val, list):
                                    algorithm_config[key] = [val]

                            tmp_config[algo_and_op_specific_field_name].extend(list(product_dict(algorithm_config)))

                op_dict[algorithm_name] = tmp_config

            hw_config.append(op_dict)

        return hw_config

    @classmethod
    def from_json(cls: type["HWConfig"], path: str) -> List[Dict[str, Any]]:
        file_path = Path(path).resolve()
        with safe_open(file_path) as f:
            json_config = json.load(f, object_pairs_hook=OrderedDict)
            return cls.from_dict(json_config)

    @staticmethod
    def get_quantization_mode_from_config_value(str_val: str) -> QuantizationMode:
        if str_val == "symmetric":
            return QuantizationMode.SYMMETRIC
        if str_val == "asymmetric":
            return QuantizationMode.ASYMMETRIC
        raise nncf.ValidationError("Invalid quantization type specified in HW config")

    @staticmethod
    def get_is_per_channel_from_config_value(str_val: str) -> bool:
        if str_val == "perchannel":
            return True
        if str_val == "pertensor":
            return False
        raise nncf.ValidationError("Invalid quantization granularity specified in HW config")

    @staticmethod
    def get_qconf_from_hw_config_subdict(quantization_subdict: Dict[str, Any]) -> QuantizerConfig:
        bits = quantization_subdict["bits"]
        mode = HWConfig.get_quantization_mode_from_config_value(quantization_subdict["mode"])
        is_per_channel = HWConfig.get_is_per_channel_from_config_value(quantization_subdict["granularity"])
        signedness_to_force = None
        if "level_low" in quantization_subdict and "level_high" in quantization_subdict:
            signedness_to_force = False
            if mode == QuantizationMode.SYMMETRIC:
                if quantization_subdict["level_low"] < 0 < quantization_subdict["level_high"]:
                    signedness_to_force = True
                true_level_low, true_level_high = quant.calculate_symmetric_level_ranges(bits, signed=True)
            else:
                signedness_to_force = True
                true_level_low, true_level_high = quant.calculate_asymmetric_level_ranges(bits)

            assert (
                quantization_subdict["level_low"] == true_level_low
            ), "Invalid value of quantizer parameter `level_low`.\
                         The parameter must be consistent with other parameters!"
            assert (
                quantization_subdict["level_high"] == true_level_high
            ), "Invalid value of quantizer parameter `level_high`.\
                         The parameter must be consistent with other parameters!"

        return QuantizerConfig(
            num_bits=bits, mode=mode, per_channel=is_per_channel, signedness_to_force=signedness_to_force
        )

    @staticmethod
    def is_qconf_list_corresponding_to_unspecified_op(qconf_list: Optional[List[QuantizerConfig]]) -> bool:
        return qconf_list is None

    @staticmethod
    def is_wildcard_quantization(qconf_list: Optional[List[QuantizerConfig]]) -> bool:
        # Corresponds to an op itself being specified in the HW config, but having no associated quantization
        # configs specified
        return qconf_list is not None and len(qconf_list) == 0

    def get_metatype_vs_quantizer_configs_map(
        self, for_weights: bool = False
    ) -> Dict[Type[OperatorMetatype], Optional[List[QuantizerConfig]]]:
        # 'None' for ops unspecified in HW config, empty list for wildcard quantization ops
        retval: Dict[Type[OperatorMetatype], Optional[List[QuantizerConfig]]] = {
            k: None for k in self._get_available_operator_metatypes_for_matching()
        }
        config_key = "weights" if for_weights else "activations"
        for op_dict in self:
            hw_config_op_name = op_dict["type"]

            metatypes = self._get_metatypes_for_hw_config_op(hw_config_op_name)
            if not metatypes:
                nncf_logger.debug(
                    "Operation name {} in HW config is not registered in NNCF under any supported operation "
                    "metatype - will be ignored".format(hw_config_op_name)
                )

            if self.QUANTIZATION_ALGORITHM_NAME in op_dict:
                allowed_qconfs = op_dict[self.QUANTIZATION_ALGORITHM_NAME].get(config_key, [])
            else:
                allowed_qconfs = []

            qconf_list_with_possible_duplicates = []
            for hw_config_qconf_dict in allowed_qconfs:
                qconf_list_with_possible_duplicates.append(self.get_qconf_from_hw_config_subdict(hw_config_qconf_dict))

            qconf_list = list(OrderedDict.fromkeys(qconf_list_with_possible_duplicates))

            for meta in metatypes:
                retval[meta] = qconf_list

        return retval

    def _get_operations_with_attribute_values(
        self, attribute_name_vs_required_value: Dict[str, Any]
    ) -> Set[Type[OperatorMetatype]]:
        result = set()
        for op_dict in self:
            if self.ATTRIBUTES_NAME not in op_dict:
                continue
            for attr_name, attr_value in attribute_name_vs_required_value.items():
                is_value_matched = op_dict[self.ATTRIBUTES_NAME].get(attr_name) == attr_value
                is_attr_set = attr_name in op_dict[self.ATTRIBUTES_NAME]
                if is_value_matched and is_attr_set:
                    hw_config_op_name = op_dict["type"]
                    metatypes = self._get_metatypes_for_hw_config_op(hw_config_op_name)
                    if not metatypes:
                        nncf_logger.debug(
                            "Operation name {} in HW config is not registered in NNCF under any supported "
                            "operation metatype - will be ignored".format(hw_config_op_name)
                        )
                    result.update(metatypes)
        return result

    def get_operations_with_unified_scales(self) -> Set[Type[OperatorMetatype]]:
        return self._get_operations_with_attribute_values({self.SCALE_ATTRIBUTE_NAME: self.UNIFIED_TYPE_NAME})

    def get_operations_with_adjusted_paddings(self) -> Set[Type[OperatorMetatype]]:
        return self._get_operations_with_attribute_values({self.ADJUST_PADDING_ATTRIBUTE_NAME: True})

    def _get_metatypes_for_hw_config_op(self, hw_config_op: HWConfigOpName) -> Set[Type[OperatorMetatype]]:
        metatypes = set()
        for op_meta in self._get_available_operator_metatypes_for_matching():
            if hw_config_op in op_meta.hw_config_names:
                metatypes.add(op_meta)
        if not metatypes:
            nncf_logger.debug(
                "Operation name {} in HW config is not registered in NNCF under any supported "
                "operation metatype - will be ignored".format(hw_config_op)
            )
        return metatypes
