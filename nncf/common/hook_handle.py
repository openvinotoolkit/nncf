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
from collections import defaultdict
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

    _registered_ids = defaultdict(set)

    def __init__(self, hooks_registry: Dict[Any, Any], mode: HookHandleIdType = HookHandleIdType.INT):
        """
        :param hooks_registry: A dictionary of hooks, indexed by hook `id`.
        :param mode: An datatype to use for the `hook_id` parameter. Default int.
        """
        self.hooks_registry_ref = weakref.ref(hooks_registry)

        if hooks_registry:
            hook_id = max(hooks_registry.keys())
            if mode == HookHandleIdType.STR:
                hook_id = str(int(hook_id) + 1)
            else:
                hook_id += 1
        else:
            hook_id = 0 if mode == HookHandleIdType.INT else "0"

        if hook_id in self.__class__._registered_ids[id(hooks_registry)]:
            raise RuntimeError("HookHandle generates non unique key")
        self.__class__._registered_ids[id(hooks_registry)].add(hook_id)
        self.hook_id = hook_id

    def remove(self):
        hooks_registry = self.hooks_registry_ref()
        if self.hook_id in self.__class__._registered_ids[id(hooks_registry)]:
            self.__class__._registered_ids[id(hooks_registry)].remove(self.hook_id)

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
            _registered_ids = defaultdict(set)

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
