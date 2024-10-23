# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional

import nncf
from nncf.common.utils.backend import BackendType


def validate_backend(data: Dict[str, Any], backend: Optional[BackendType]) -> None:
    """
    Checks whether backend in loaded data is equal to a provided backend.

    :param data: Loaded statistics.
    :param backend: Provided backend.
    """
    if "backend" not in data:
        raise nncf.ValidationError("The provided metadata has no information about backend.")
    b = data["backend"]
    if data["backend"] != backend.value:
        raise nncf.ValidationError(
            f"Backend in loaded statistics {b} does not match to an expected backend {backend.value}."
        )
