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

import gc
from collections import OrderedDict

import pytest

from nncf.common.hook_handle import HookHandle
from nncf.common.hook_handle import HookHandleIdType


@pytest.mark.parametrize("id_type, cls_", [(HookHandleIdType.INT, int), (HookHandleIdType.STR, str)])
def test_hook_handle_adds_removes_unique_keys(id_type: HookHandleIdType, cls_):
    num_hooks = 3
    hook_registry = OrderedDict()
    handles = [HookHandle(hook_registry, id_type) for _ in range(num_hooks)]
    for idx, handle in enumerate(handles):
        handle.add(idx)
    assert len(hook_registry) == num_hooks

    assert list(hook_registry.keys()) == list(map(cls_, range(num_hooks)))
    assert [h.hook_id for h in handles] == list(map(cls_, range(num_hooks)))

    handles[0].remove()
    assert list(hook_registry.keys()) == list(map(cls_, range(1, num_hooks)))


def test_two_hooks_one_handle_error():
    hook_registry = OrderedDict()
    handle = HookHandle(hook_registry)
    handle.add(0)
    with pytest.raises(RuntimeError):
        handle.add(1)


def test_hook_id_getter_before_init():
    hook_registry = OrderedDict()
    handle = HookHandle(hook_registry)
    with pytest.raises(RuntimeError):
        handle.hook_id


def test_remove_before_init():
    hook_registry = OrderedDict()
    handle = HookHandle(hook_registry)
    with pytest.raises(RuntimeError):
        handle.remove()


def test_handle_does_not_fail_if_hook_does_not_exist():
    hook_registry = OrderedDict()
    handle = HookHandle(hook_registry)
    handle.add(0)
    del hook_registry[handle.hook_id]
    handle.remove()
    del hook_registry
    handle.remove()


def test_hook_handle_weak_ref():
    def _local_scope_fn():
        hook_registry = OrderedDict()
        handle = HookHandle(hook_registry)
        handle.add(1)
        return handle

    handle = _local_scope_fn()
    gc.collect()
    assert handle.hooks_registry_ref() is None
