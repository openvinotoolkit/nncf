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

import weakref
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


class WeakUnhashableKeyMap(Generic[_K, _V]):
    """
    A dictionary-like object that uses weak references to unhashable objects as keys.

    This class allows the use of unhashable objects as keys in a dictionary by using their `id`.
    It ensures that the entries are removed when the objects are garbage collected.

    :param _data: Internal storage for the key-value pairs,
        where the key is the id of the object and the value is a tuple containing a weak reference
        to the object and the actual value.
    """

    def __init__(self) -> None:
        """Initialize an empty WeakUnhashableKeyMap."""
        self._data: Dict[int, Tuple[weakref.ReferenceType[_K], _V]] = {}

    def __getitem__(self, obj: Any) -> Any:
        """
        Get the value associated with the given object.

        :param obj (Any): The key object.
        :return: The value associated with the object.
        """
        ref_obj, val = self._data[id(obj)]
        if ref_obj() is not obj:
            raise KeyError(obj)
        return val

    def __setitem__(self, obj: _K, value: _V) -> None:
        """
        Set the value for the given object.

        :param obj: The key object.
        :param value: The value to associate with the object.
        """
        key = id(obj)
        ref_obj, _ = self._data.get(key, (None, None))

        if ref_obj is not None and ref_obj() is obj:
            self._data[key] = ref_obj, value
        else:

            def on_destroy(_: Any) -> None:
                del self._data[key]

            self._data[key] = weakref.ref(obj, on_destroy), value

    def get(self, obj: _K, default: Optional[Any] = None) -> Optional[_V]:
        """
        Get the value associated with the given object, or a default value if the object is not in the map.

        :param obj: The key object.
        :param default: The default value to return if the object is not found. Defaults to None.
        :return: The value associated with the object, or the default value if the object is not found.
        """
        key = id(obj)
        try:
            _, val = self._data[key]
            return val
        except KeyError:
            return default
