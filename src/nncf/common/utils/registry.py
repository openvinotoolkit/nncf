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

from typing import Any, Callable, TypeVar

TClass = TypeVar("TClass", bound=type)


class Registry:
    REGISTERED_NAME_ATTR = "_registered_name"

    def __init__(self, name: str, add_name_as_attr: bool = False):
        self._name = name
        self._registry_dict: dict[str, Any] = {}
        self._add_name_as_attr = add_name_as_attr

    @property
    def registry_dict(self) -> dict[str, Any]:
        return self._registry_dict

    def values(self) -> Any:
        return self._registry_dict.values()

    def _register(self, obj: Any, name: str) -> None:
        if name in self._registry_dict:
            msg = f"{name} is already registered in {self._name}"
            raise KeyError(msg)
        self._registry_dict[name] = obj

    def register(self, name: str = None) -> Callable[[TClass], TClass]:
        def wrap(obj: TClass) -> TClass:
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, name)
            self._register(obj, cls_name)
            return obj

        return wrap

    def get(self, name: str) -> Any:
        if name not in self._registry_dict:
            self._key_not_found(name)
        return self._registry_dict[name]

    def _key_not_found(self, name: str) -> None:
        msg = f"{name} is unknown type of {self._name} "
        raise KeyError(msg)

    def __contains__(self, item: Any) -> bool:
        return item in self._registry_dict.values()
