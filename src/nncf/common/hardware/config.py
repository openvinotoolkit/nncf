# Copyright (c) 2026 Intel Corporation
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
from typing import Any, Optional

import nncf
from nncf import TargetDevice
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.hardware.defines import ADJUST_PADDING
from nncf.common.hardware.defines import SCALES
from nncf.common.hardware.defines import UNIFIED
from nncf.common.hardware.defines import OpDesc
from nncf.common.hardware.setups.cpu import CPU_SETUP
from nncf.common.hardware.setups.gpu import GPU_SETUP
from nncf.common.hardware.setups.npu import NPU_SETUP
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizerConfig


def get_hw_setup(target_device: TargetDevice) -> tuple[OpDesc, ...]:
    """
    Retrieves the hardware setup configuration based on the specified target device.

    :param target_device: The target device for which to retrieve the hardware setup.
    :return: A list of operation descriptors corresponding to the hardware setup.
    """
    if target_device in [TargetDevice.ANY, TargetDevice.CPU, TargetDevice.CPU_SPR]:
        return CPU_SETUP
    if target_device == TargetDevice.GPU:
        return GPU_SETUP
    if target_device == TargetDevice.NPU:
        return NPU_SETUP
    msg = f"Unsupported target device: {target_device}"
    raise nncf.InternalError(msg)


class HWConfig(ABC):
    """
    This class provides an interface for managing hardware-specific configurations
    related to quantization and operator metatypes. It allows for the retrieval of
    available operator metatypes, quantizer configurations, and operations with
    specific attribute values.

    :param hw_setup: A tuple of hardware setup descriptors for the target device.
    :param target_device: The target device for which the hardware configuration is set.
    """

    def __init__(self, target_device: TargetDevice) -> None:
        self.hw_setup = get_hw_setup(target_device)
        self.target_device = target_device

    @abstractmethod
    def _get_available_operator_metatypes_for_matching(self) -> list[type[OperatorMetatype]]:
        """
        Retrieve a list of available backend specific operator metatypes.

        :return: A list of operator metatypes available for matching.
        """
        pass

    def get_metatype_vs_quantizer_configs_map(
        self, for_weights: bool = False
    ) -> dict[type[OperatorMetatype], Optional[list[QuantizerConfig]]]:
        """
        Retrieves a mapping of operator metatypes to their corresponding quantizer configurations.

        :param for_weights: A flag indicating whether to retrieve quantizer configurations for weights.
        :return: A dictionary mapping operator metatypes to their corresponding quantizer configurations.
        """
        # 'None' for ops unspecified in HW config, empty list for wildcard quantization ops
        retval: dict[type[OperatorMetatype], Optional[list[QuantizerConfig]]] = {
            k: None for k in self._get_available_operator_metatypes_for_matching()
        }
        for op_desc in self.hw_setup:
            hw_config_op_name = op_desc.type
            metatypes = self._get_metatypes_for_hw_config_op(hw_config_op_name)
            allowed_spaces = op_desc.weights if for_weights else op_desc.activations

            # Deduplication
            # There are no duplicates in existing HW configs, but this deduplication is just in case of
            # a custom HW config with duplicates
            qconf_list_with_possible_duplicates = []
            for q_space in allowed_spaces:
                qconf_list_with_possible_duplicates.extend(q_space.get_all_qconfigs())
            qconf_list = list(dict.fromkeys(qconf_list_with_possible_duplicates))

            for meta in metatypes:
                retval[meta] = qconf_list

        return retval

    def _get_metatypes_for_hw_config_op(self, hw_config_op: str) -> set[type[OperatorMetatype]]:
        """
        Retrieve a set of operator metatypes that match the given hardware operation name.

        :param hw_config_op: The hardware configuration operation name to match against.
        :return: A set of operator metatypes that correspond to the specified hardware configuration operation.
        """
        metatypes = set()
        for op_meta in self._get_available_operator_metatypes_for_matching():
            if hw_config_op in op_meta.hw_config_names:
                metatypes.add(op_meta)
        return metatypes

    @staticmethod
    def is_qconf_list_corresponding_to_unspecified_op(qconf_list: Optional[list[QuantizerConfig]]) -> bool:
        """
        Check if the provided quantizer configuration list corresponds to an unspecified operation.

        This function determines if the given list of quantizer configurations is None,
        which indicates that there is no specific quantization configuration for the operation.

        :param qconf_list: A list of quantizer configurations or None.
        :return: True if the list is None, indicating an unspecified operation; False otherwise.
        """
        return qconf_list is None

    @staticmethod
    def is_wildcard_quantization(qconf_list: Optional[list[QuantizerConfig]]) -> bool:
        """
        Determines if the provided list of quantizer configurations represents a wildcard quantization.

        A wildcard quantization occurs when the hardware configuration specifies an operation
        but does not provide any associated quantization configurations.

        :param qconf_list: A list of quantizer configurations or None.
        :return: True if the list is not None and is empty, indicating a wildcard quantization; False otherwise.
        """
        return qconf_list is not None and len(qconf_list) == 0

    def _get_operations_with_attribute_values(
        self, attribute_name_vs_required_value: dict[str, Any]
    ) -> set[type[OperatorMetatype]]:
        """
        Retrieves a set of operation metatypes that match the specified attribute values.

        :param attribute_name_vs_required_value: A attribute dictionary.
        :return: A set of operation metatypes that match the specified attribute values.
        """
        result = set()
        for op_desc in self.hw_setup:
            for attr_name, attr_value in attribute_name_vs_required_value.items():
                is_value_matched = op_desc.attributes.get(attr_name) == attr_value
                is_attr_set = attr_name in op_desc.attributes
                if is_value_matched and is_attr_set:
                    hw_config_op_name = op_desc.type
                    metatypes = self._get_metatypes_for_hw_config_op(hw_config_op_name)
                    if not metatypes:
                        nncf_logger.debug(
                            f"Operation name {hw_config_op_name} in HW config is not registered in NNCF"
                            " under any supported operation metatype - will be ignored"
                        )
                    result.update(metatypes)
        return result

    def get_operations_with_unified_scales(self) -> set[type[OperatorMetatype]]:
        """
        Retrieves a set of operations that have unified scale attributes.

        :return: A set of operations metatypes.
        """
        return self._get_operations_with_attribute_values({SCALES: UNIFIED})

    def get_operations_with_adjusted_paddings(self) -> set[type[OperatorMetatype]]:
        """
        Retrieves a set of operations that have adjust padding attributes.

        :return: A set of operations metatypes.
        """
        return self._get_operations_with_attribute_values({ADJUST_PADDING: True})
