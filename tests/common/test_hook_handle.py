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

import gc
from collections import OrderedDict

from nncf.common.hook_handle import add_op_to_registry


def test_hook_handle_adds_removes_unique_keys():
    num_hooks = 3
    hook_registry = OrderedDict()
    handles = []
    for _ in range(num_hooks):
        handles.append(add_op_to_registry(hook_registry, 0))

    assert list(hook_registry.keys()) == list(map(str, range(num_hooks)))
    assert [h.hook_id for h in handles] == list(map(str, range(num_hooks)))

    handles[0].remove()
    assert list(hook_registry.keys()) == list(map(str, range(1, num_hooks)))


def test_handle_does_not_fail_if_hook_does_not_exist():
    hook_registry = OrderedDict()
    handle = add_op_to_registry(hook_registry, 0)
    del hook_registry[handle.hook_id]
    handle.remove()
    del hook_registry
    handle.remove()


def test_handle_does_not_remove_op_twice():
    hook_registry = OrderedDict()
    handle = add_op_to_registry(hook_registry, 0)
    handle.remove()
    assert not hook_registry

    second_handle = add_op_to_registry(hook_registry, 1)
    assert handle.hook_id == second_handle.hook_id
    assert "0" in hook_registry

    handle.remove()
    assert "0" in hook_registry
    assert hook_registry["0"] == 1


def test_hook_handle_weak_ref():
    def _local_scope_fn():
        hook_registry = OrderedDict()
        return add_op_to_registry(hook_registry, 0)

    handle = _local_scope_fn()
    gc.collect()
    assert handle.hooks_registry_ref() is None
