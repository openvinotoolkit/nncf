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
from typing import Dict

import pytest

from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.hardware.config import HWConfig
from nncf.common.hardware.config import get_hw_config_type
from nncf.parameters import TargetDevice

BASE_DEVICE = TargetDevice.CPU


def load_config_for_device(target_device: TargetDevice) -> HWConfig:
    """
    Loads hardware configuration based on the device.

    :param target_device: TargetDevice instance.
    :return: Hardware configuration as dictionary.
    """
    hw_config_type = get_hw_config_type(target_device)
    hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
    return HWConfig.from_json(hw_config_path)


def get_quantization_config(hw_config: HWConfig) -> Dict:
    """
    Returns quantization config aggregated by types.

    :param hw_config: HWConfig instance.
    :return: Dictionary with the configuration by types.
    """
    return {c["type"]: c["quantization"] for c in hw_config if "quantization" in c}


@pytest.mark.parametrize("target_device", [TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU])
def test_get_hw_config_type(target_device):
    expected = HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device.value]
    mesured = get_hw_config_type(target_device.value)
    assert expected == mesured.value


def test_get_hw_config_type_trial():
    assert get_hw_config_type("TRIAL") is None


@pytest.mark.parametrize("target_device", [TargetDevice.NPU])
def test_device_configuration_alignment(target_device):
    base_hw_config = load_config_for_device(BASE_DEVICE)
    base_quantization_config = get_quantization_config(base_hw_config)

    test_hw_config = load_config_for_device(target_device)
    test_quantization_config = get_quantization_config(test_hw_config)

    for layer_type, layer_configs in base_quantization_config.items():
        assert layer_type in test_quantization_config, f"{layer_type} was not found in test configuration"
        test_layer_config = test_quantization_config[layer_type]
        for config_type, type_options in layer_configs.items():
            assert (
                config_type in test_layer_config
            ), f"{config_type} was not found in test configuration for {layer_type}"
            test_type_options = test_layer_config[config_type]
            for idx, option in enumerate(type_options):
                assert (
                    option == test_type_options[idx]
                ), f"#{idx} option was not aligned for {config_type} type in {layer_type}"
