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

from typing import Any, Callable, TypeVar, Union

TObj = TypeVar("TObj", bound=Union[Callable[..., Any], type])

API_MARKER_ATTR = "_nncf_api_marker"
CANONICAL_ALIAS_ATTR = "_nncf_canonical_alias"


def api(canonical_alias: str = None) -> Callable[[TObj], TObj]:
    """
    Decorator function used to mark a object as an API.

    Example:
        @api(canonical_alias="alias")
        class Class:
            pass

        @api(canonical_alias="alias")
        def function():
            pass

    :param canonical_alias: The canonical alias for the API class.
    """

    def decorator(obj: TObj) -> TObj:
        setattr(obj, API_MARKER_ATTR, obj.__name__)
        if canonical_alias is not None:
            setattr(obj, CANONICAL_ALIAS_ATTR, canonical_alias)
        return obj

    return decorator
