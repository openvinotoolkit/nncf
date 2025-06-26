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

from nncf.tensorflow.graph.utils import get_original_name_and_instance_idx


def is_ignored(node_name, ignored_scopes):
    original_name, _ = get_original_name_and_instance_idx(node_name)
    return any(
        (
            re.fullmatch(ignored.replace("{re}", ""), original_name)
            if ignored.startswith("{re}")
            else ignored == original_name
        )
        for ignored in ignored_scopes
    )
