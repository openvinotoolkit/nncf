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

from nncf.common.graph.patterns import HWFusedPatternNames
from nncf.common.graph.patterns import IgnoredPatternNames
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.utils.backend import BackendType


def check_hw_patterns(backend: BackendType, reasons: Dict[HWFusedPatternNames, str]):
    backend_patterns = PatternsManager._get_backend_hw_patterns_map(backend)

    all_base_patterns = HWFusedPatternNames
    for base_pattern in all_base_patterns:
        pattern_name = base_pattern.name
        if base_pattern in reasons:
            assert base_pattern not in backend_patterns, f"Pattern {pattern_name} found in {backend.name}"
            ignore_reason = reasons[base_pattern]
            print(f"{pattern_name} is ignored. Reason: {ignore_reason}")
            continue
        assert base_pattern in backend_patterns, f"Pattern {pattern_name} not found in {backend.name}"


def check_ignored_patterns(backend: BackendType, reasons: Dict[IgnoredPatternNames, str]):
    backend_patterns = PatternsManager._get_backend_ignored_patterns_map(backend)

    all_base_patterns = IgnoredPatternNames
    for base_pattern in all_base_patterns:
        pattern_name = base_pattern.name
        if base_pattern in reasons:
            assert base_pattern not in backend_patterns, f"Pattern {pattern_name} found in {backend.name}"
            ignore_reason = reasons[base_pattern]
            print(f"{pattern_name} is ignored. Reason: {ignore_reason}")
            continue
        assert base_pattern in backend_patterns, f"Pattern {pattern_name} not found in {backend.name}"
