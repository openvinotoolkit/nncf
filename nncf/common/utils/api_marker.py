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
from typing import Any


class api:
    API_MARKER_ATTR = "_nncf_api_marker"
    CANONICAL_ALIAS_ATTR = "_nncf_canonical_alias"

    def __init__(self, canonical_alias: str = None):
        self._canonical_alias = canonical_alias

    def __call__(self, obj: Any) -> Any:
        # The value of the marker will be useful in determining
        # whether we are handling a base class or a derived one.
        setattr(obj, api.API_MARKER_ATTR, obj.__name__)
        if self._canonical_alias is not None:
            setattr(obj, api.CANONICAL_ALIAS_ATTR, self._canonical_alias)
        return obj


def is_api(obj: Any) -> bool:
    return hasattr(obj, api.API_MARKER_ATTR)
