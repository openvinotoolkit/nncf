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


def is_experimental_torch_tracing_enabled() -> bool:
    """
    Checks if experimental torch tracing is enabled by environment variable NNCF_EXPERIMENTAL_TORCH_TRACING.

    :return: True if experimental torch tracing is enabled, False otherwise.
    """
    return os.getenv("NNCF_EXPERIMENTAL_TORCH_TRACING") is not None
