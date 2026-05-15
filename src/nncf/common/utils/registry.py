# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Generic, TypeVar, ValuesView, cast

TKey = TypeVar("TKey")
TObject = TypeVar("TObject")
TRegisterObject = TypeVar("TRegisterObject")
# Used another TRegisterObject for register method, to avoid replace type of registered object.
# It may impossible to check type of registered object.
# PEP695 provide solution for this problem, but it is not supported in python <3.12


class Registry(Generic[TKey, TObject]):
    REGISTERED_NAME_ATTR = "_registered_name"

    def __init__(self, name: str, add_name_as_attr: bool = False):
        self._name = name
        self._registry_dict: dict[TKey, TObject] = {}
        self._add_name_as_attr = add_name_as_attr

    @property
    def registry_dict(self) -> dict[TKey, TObject]:
        return self._registry_dict

    def values(self) -> ValuesView[TObject]:
        return self._registry_dict.values()

    def _register(self, obj: TObject, name: TKey) -> None:
        if name in self._registry_dict:
            msg = f"{name} is already registered in {self._name}"
            raise KeyError(msg)
        self._registry_dict[name] = obj

    def register(self, name: TKey | None = None) -> Callable[[TRegisterObject], TRegisterObject]:
        def wrap(obj: TRegisterObject) -> TRegisterObject:
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, cls_name)
            self._register(cast(TObject, obj), cls_name)
            return obj

        return wrap

    def get(self, name: TKey) -> TObject:
        if name not in self._registry_dict:
            self._key_not_found(name)
        return self._registry_dict[name]

    def _key_not_found(self, name: TKey) -> None:
        msg = f"{name} is unknown type of {self._name} "
        raise KeyError(msg)

    def __contains__(self, item: TObject) -> bool:
        return item in self._registry_dict.values()
