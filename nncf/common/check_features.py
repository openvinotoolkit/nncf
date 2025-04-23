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

import os


def is_torch_tracing_by_patching() -> bool:
    """
    Checks if legacy torch tracing is enabled by environment variable NNCF_TORCH_LEGACY_TRACING.

    True - will wrap model by NNCFNetwork and patch function in torch namespace.
    False - will use FunctionHookMode without patching torch namespace.

    :return: True if legacy torch tracing is enabled, False otherwise.
    """
    return os.getenv("NNCF_TORCH_LEGACY_TRACING", "").lower() in ["1", "on", "true"]
