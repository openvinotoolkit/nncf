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
from typing import Any, Dict, Optional, Union


class HookHandle:
    """
    A handle to remove a hook. Does not guarantee that
    hook_id is unique for the hooks_registry.
    """

    def __init__(self, hooks_registry: Dict[Any, Any], hook_id: str):
        """
        :param hooks_registry: A dictionary of hooks, indexed by hook `id`.
        :param hook_id: Hook id to use as key in the dictionary of hooks.
        """
        self.hooks_registry_ref: Optional[weakref.ReferenceType[Dict[Any, Any]]] = weakref.ref(hooks_registry)
        self._hook_id = hook_id

    @property
    def hook_id(self) -> Union[int, str]:
        """
        Key to use to retrieve handle from the registry.
        """
        return self._hook_id

    def remove(self) -> None:
        """
        Removes the corresponding operation from the registered hooks registry if it is possible.
        """
        if self.hooks_registry_ref is None:
            return

        hooks_registry = self.hooks_registry_ref()
        if hooks_registry is not None and self.hook_id in hooks_registry:
            del hooks_registry[self.hook_id]
            self.hooks_registry_ref = None


def add_op_to_registry(hooks_registry: Dict[Any, Any], op: Any) -> HookHandle:
    """
    Registers op into the hooks_registry and returns HookHandler instance.

    :param hooks_registry: A dictionary of hooks, indexed by hook `id`.
    :param op: Operation to set to registered hooks registry.
    :return: HookHandle that contains the registry of hooks and
        the id of operation to remove it from the registry.
    """
    if hooks_registry:
        hook_id = max(map(int, hooks_registry))
        hook_id_str = str(hook_id + 1)
    else:
        hook_id_str = "0"
    hooks_registry[hook_id_str] = op
    return HookHandle(hooks_registry, hook_id_str)
