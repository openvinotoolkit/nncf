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

from typing import Callable, Generic, TypeVar, ValuesView

TKey = TypeVar("TKey")
TObject = TypeVar("TObject")
TRegisterObject = TypeVar("TRegisterObject")
# Separate type variable preserves input type of decorated object.
# PEP695 can express a stricter relation to TObject, but it is unavailable in Python < 3.12.


class Registry(Generic[TKey, TObject]):
    """
    Generic key-to-object registry.

    Stores objects by key and provides a decorator-based registration API.
    """

    REGISTERED_NAME_ATTR = "_registered_name"

    def __init__(self, name: str, add_name_as_attr: bool = False):
        """
        :param name: Human-readable registry name used in error messages.
        :param add_name_as_attr: Whether to set the registered key on object attribute
            `REGISTERED_NAME_ATTR`.
        """
        self._name = name
        self._registry_dict: dict[TKey, TObject] = {}
        self._add_name_as_attr = add_name_as_attr

    @property
    def registry_dict(self) -> dict[TKey, TObject]:
        """
        Return the underlying registry mapping.

        :return: Dictionary with registered objects.
        """
        return self._registry_dict

    def values(self) -> ValuesView[TObject]:
        """
        Return registered object values.

        :return: View over registered objects.
        """
        return self._registry_dict.values()

    def _register(self, obj: TObject, name: TKey) -> None:
        if name in self._registry_dict:
            msg = f"{name} is already registered in {self._name}"
            raise KeyError(msg)
        self._registry_dict[name] = obj

    def register(self, name: TKey | None = None) -> Callable[[TRegisterObject], TRegisterObject]:
        """
        Create a decorator that registers an object in the registry.

        If `name` is not provided, `obj.__name__` is used.

        :param name: Explicit key for registration.
        :return: Decorator that registers and returns the input object.
        """

        def wrap(obj: TRegisterObject) -> TRegisterObject:
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__  # type: ignore[attr-defined]
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, cls_name)
            self._register(obj, cls_name)  # type: ignore[arg-type]
            return obj

        return wrap

    def get(self, name: TKey) -> TObject:
        """
        Get a registered object by key.

        :param name: Registry key.
        :return: Registered object associated with `name`.
        """
        if name not in self._registry_dict:
            self._key_not_found(name)
        return self._registry_dict[name]

    def _key_not_found(self, name: TKey) -> None:
        msg = f"{name} is unknown type of {self._name} "
        raise KeyError(msg)

    def __contains__(self, item: TObject) -> bool:
        """
        Check whether an object instance is present in registered values.

        :param item: Object to check.
        :return: `True` if object is registered, otherwise `False`.
        """
        return item in self._registry_dict.values()
