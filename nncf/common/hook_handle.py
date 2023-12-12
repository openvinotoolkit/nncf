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
from typing import Any, Dict


class HookHandleIdType(Enum):
    """
    Enum of possible types for the `HookHandle.id`.
    """

    INT = 0
    STR = 1


class HookHandle:
    """
    A handle to remove a hook.
    """

    id = 0

    def __init__(self, hooks_registry: Dict[Any, Any], mode: HookHandleIdType = HookHandleIdType.INT):
        """
        :param hooks_registry: A dictionary of hooks, indexed by hook `id`.
        :param mode: An datatype to use for the `hook_id` parameter. Default int.
        """
        self.hooks_registry_ref = weakref.ref(hooks_registry)
        self.hook_id = self.__class__.id
        if mode == HookHandleIdType.STR:
            self.hook_id = str(self.hook_id)
        self.__class__.id += 1

    def remove(self):
        hooks_registry = self.hooks_registry_ref()
        if hooks_registry is not None and self.hook_id in hooks_registry:
            del hooks_registry[self.hook_id]


class HookHandleManager:
    """
    Hook handle manager which creates local HookHandle class with
    id counter unique to the HookHandleManeger instance.
    This allows to use local id counters in each
    individual NNCF module / NNCFNetwork, so hook ids are not depent on
    global HookHandle id state but unique for each individual NNCF module / NNCFNetwork.
    """

    def __init__(self) -> None:
        class LocalHookHandle(HookHandle):
            id = 0

        self._hook_handle = LocalHookHandle

    def create_handle(
        self, hooks_registry: Dict[Any, Any], mode: HookHandleIdType = HookHandleIdType.INT
    ) -> HookHandle:
        """
        Creates an instance of the HookHandler with id counter unique to this HookHandleManager.

        :param hooks_registry: A dictionary of hooks, indexed by hook `id`.
        :param mode: An datatype to use for the `hook_id` parameter. Default int.
        """
        return self._hook_handle(hooks_registry=hooks_registry, mode=mode)
