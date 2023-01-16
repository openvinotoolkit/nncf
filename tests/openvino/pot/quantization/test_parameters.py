"""
 Copyright (c) 2022 Intel Corporation
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
from openvino.tools.pot.algorithms.quantization.utils import \
    load_hardware_config

from nncf.parameters import TargetDevice


@pytest.mark.parametrize('target_device', TargetDevice)
def test_target_device(target_device):
    config = {'target_device': target_device.value}
    assert load_hardware_config(config)
