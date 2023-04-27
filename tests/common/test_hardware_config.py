"""
 Copyright (c) 2023 Intel Corporation
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
import pytest

from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.hardware.config import get_hw_config_type
from nncf.parameters import TargetDevice


@pytest.mark.parametrize("target_device", [TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU, TargetDevice.VPU])
def test_get_hw_config_type(target_device):
    expected = HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device.value]
    mesured = get_hw_config_type(target_device.value)
    assert expected == mesured.value


def test_get_hw_config_type_trial():
    assert get_hw_config_type("TRIAL") is None


def test_get_hw_config_type_cpu_spr():
    with pytest.raises(ValueError):
        get_hw_config_type("CPU_SPR")
