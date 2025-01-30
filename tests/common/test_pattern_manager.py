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

from enum import Enum

from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.patterns import PatternDesc
from nncf.common.utils.registry import Registry
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice

TEST_DEVICE_PATTERN_REGISTRY = Registry("TEST_PATTERNS_REGISTRY")


# pylint: disable=protected-access


class DevicePatterns(Enum):
    CPU_PATTERN = PatternDesc("CPU_PATTERN", devices=[TargetDevice.CPU])
    GPU_PATTERN = PatternDesc("GPU_PATTERN", devices=[TargetDevice.GPU])
    COMMON_PATTERN = PatternDesc("COMMON_PATTERN")


TEST_DEVICE_PATTERN_REGISTRY.register(DevicePatterns.CPU_PATTERN)(None)
TEST_DEVICE_PATTERN_REGISTRY.register(DevicePatterns.GPU_PATTERN)(None)
TEST_DEVICE_PATTERN_REGISTRY.register(DevicePatterns.COMMON_PATTERN)(None)


def test_pattern_filter_device():
    manager = PatternsManager()
    filtered_patterns = manager._filter_patterns(
        TEST_DEVICE_PATTERN_REGISTRY.registry_dict, device=TargetDevice.CPU, model_type=None
    )
    assert len(filtered_patterns) == 2
    assert DevicePatterns.CPU_PATTERN in filtered_patterns
    assert DevicePatterns.COMMON_PATTERN in filtered_patterns


TEST_MODEL_TYPE_PATTERN_REGISTRY = Registry("TEST_PATTERNS_REGISTRY")


class ModelTypePatterns(Enum):
    TRANSFORMER_PATTERN = PatternDesc("TRANSFORMER_PATTERN", model_types=[ModelType.TRANSFORMER])
    COMMON_PATTERN = PatternDesc("COMMON_PATTERN")


TEST_MODEL_TYPE_PATTERN_REGISTRY.register(ModelTypePatterns.TRANSFORMER_PATTERN)(None)
TEST_MODEL_TYPE_PATTERN_REGISTRY.register(ModelTypePatterns.COMMON_PATTERN)(None)


def test_pattern_filter_model_type():
    manager = PatternsManager()
    filtered_patterns = manager._filter_patterns(
        TEST_MODEL_TYPE_PATTERN_REGISTRY.registry_dict, device=None, model_type=None
    )
    assert len(filtered_patterns) == 1
    assert ModelTypePatterns.COMMON_PATTERN in filtered_patterns

    filtered_patterns = manager._filter_patterns(
        TEST_MODEL_TYPE_PATTERN_REGISTRY.registry_dict, device=None, model_type=ModelType.TRANSFORMER
    )
    assert len(filtered_patterns) == 2
    assert ModelTypePatterns.COMMON_PATTERN in filtered_patterns
    assert ModelTypePatterns.TRANSFORMER_PATTERN in filtered_patterns
