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

import re

import openvino as ov

_IS_ARM_CPU = None
_IS_LNL_CPU = None


def _get_cpu_name() -> str:
    """
    :return: The name of the CPU.
    """
    return ov.Core().get_property("CPU", ov.properties.device.full_name)


def is_arm_cpu() -> bool:
    """
    Checks whether current CPU is an ARM CPU or not.
    :return: True if current CPU is an ARM CPU, False otherwise.
    """
    global _IS_ARM_CPU
    if _IS_ARM_CPU is None:
        _IS_ARM_CPU = "arm" in _get_cpu_name().lower()
    return _IS_ARM_CPU


def is_lnl_cpu() -> bool:
    """
    Checks whether current CPU is an Intel Lunar Lake generation or not.
    :return: True if current CPU is an Intel Lunar Lake generation, False otherwise.
    """
    global _IS_LNL_CPU
    if _IS_LNL_CPU is None:
        _IS_LNL_CPU = re.search(r"Ultra \d 2\d{2}", _get_cpu_name()) is not None
    return _IS_LNL_CPU
