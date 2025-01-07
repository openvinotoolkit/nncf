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
from typing import List


def should_consider_scope(scope_str: str, target_scopes: List[str], ignored_scopes: List[str]):
    # TODO: rewrite and add target_scopes handling
    return all(
        not re.fullmatch(ignored.replace("{re}", ""), scope_str) if ignored.startswith("{re}") else scope_str != ignored
        for ignored in ignored_scopes
    )
