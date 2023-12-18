# Copyright (c) 2023 Intel Corporation
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
from enum import Enum
from typing import Any, Dict, Union


class HookHandleIdType(Enum):
    """
    Enum of possible types for the HookHandle hook_id.
    """

    INT = 0
    STR = 1


class HookHandle:
    """
    A handle to remove a hook.
    """

    def __init__(self, hooks_registry: Dict[Any, Any], mode: HookHandleIdType = HookHandleIdType.INT):
        """
        :param hooks_registry: A dictionary of hooks, indexed by hook `id`.
        :param mode: An datatype to use for the `hook_id` parameter. Default int.
        """
        self.hooks_registry_ref = weakref.ref(hooks_registry)
        self._mode = mode
        self._hook_id = None
        self._op_registered = False

    @property
    def hook_id(self) -> Union[int, str]:
        if self._hook_id is None:
            raise RuntimeError("Attempt to use HookHandle hook id before actual hook registration.")
        return self._hook_id

    def add(self, op: Any) -> None:
        """
        Adds operation to registered hooks registry.

        :param op: Operation to set to registered hooks registry.
        """
        if self._hook_id is not None:
            raise RuntimeError("Attempt to use one HookHandle for two hooks.")

        hooks_registry = self.hooks_registry_ref()
        if hooks_registry:
            hook_id = max(hooks_registry.keys())
            if self._mode == HookHandleIdType.STR:
                hook_id = str(int(hook_id) + 1)
            else:
                hook_id += 1
        else:
            hook_id = 0 if self._mode == HookHandleIdType.INT else "0"
        self._hook_id = hook_id
        hooks_registry[self._hook_id] = op

    def remove(self) -> None:
        """
        Removes added operation from registered hooks registry if it is possible.
        """
        hooks_registry = self.hooks_registry_ref()
        if hooks_registry is not None and self.hook_id in hooks_registry:
            del hooks_registry[self.hook_id]
