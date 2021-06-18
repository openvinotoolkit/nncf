"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import weakref


class HookHandle:
    """
    A handle to remove a hook
    """
    id = 0

    def __init__(self, hooks_registry):
        self.hooks_registry_ref = weakref.ref(hooks_registry)
        self.hook_id = HookHandle.id
        HookHandle.id = HookHandle.id + 1

    def remove(self):
        hooks_registry = self.hooks_registry_ref()
        if hooks_registry is not None and self.hook_id in hooks_registry:
            del hooks_registry[self.hook_id]
