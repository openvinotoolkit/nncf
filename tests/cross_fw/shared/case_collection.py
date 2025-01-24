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

from typing import Dict, List

import pytest

COMMON_SCOPE_MARKS_VS_OPTIONS = {
    # for instance
    # "slow": "--run-slow",
}


def skip_marked_cases_if_options_not_specified(config, items, marks_vs_options: Dict[str, str]) -> None:
    options_not_given = {mark: option for mark, option in marks_vs_options.items() if config.getoption(option) is None}
    for item in items:
        for mark, option in options_not_given.items():
            if mark in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason=f"This test case requires an option {option} to be specified for pytest.")
                )


def skip_if_backend_not_selected(backend: str, backends_list: List[str]):
    if "all" not in backends_list and backend not in backends_list:
        pytest.skip("not selected for testing")
