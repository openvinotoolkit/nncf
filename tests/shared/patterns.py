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
from typing import Dict

from nncf.common.utils.backend import BackendType
from nncf.common.graph.patterns import PatternNames
from nncf.common.graph.patterns.manager import PatternsManager


def check_patterns(backend: BackendType, reasons: Dict[PatternNames, str]):
    backend_patterns = PatternsManager.get_backend_patterns_map(backend)

    all_base_apatterns = PatternNames
    for base_pattern in all_base_apatterns:
        if base_pattern in reasons:
            ignore_reason = reasons[base_pattern]
            print(f'{base_pattern.name} is ignored. Reason: {ignore_reason}')
            continue
        assert base_pattern in backend_patterns, f'Pattern {base_pattern.name} not found in {backend.name}'
