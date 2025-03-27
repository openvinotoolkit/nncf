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

from typing import List

import pytest
import torch
from torch import nn

from nncf.experimental.torch2.function_hook.hook_storage import HookStorage
from nncf.experimental.torch2.function_hook.hook_storage import unwrap_hook_name
from tests.torch2.function_hook.helpers import CallCount


@pytest.fixture(params=["pre_hook", "post_hook"])
def hook_type(request: pytest.FixtureRequest) -> str:
    return request.param


def test_insert():
    hook_storage = HookStorage()
    hook = nn.Identity()
    hook_storage.register_pre_function_hook("foo", 0, hook)
    assert hook_storage.pre_hooks["foo__0"]["0"] is hook

    hook_storage.register_post_function_hook("foo", 0, hook)
    assert hook_storage.post_hooks["foo__0"]["0"] is hook


def test_execute():
    hook_storage = HookStorage()

    pre_hook = CallCount()
    hook_storage.register_pre_function_hook("foo", 0, pre_hook)
    hook_storage.execute_pre_function_hooks("foo", 0, None)
    assert pre_hook.call_count == 1

    post_hook = CallCount()
    hook_storage.register_post_function_hook("foo", 0, post_hook)
    hook_storage.execute_post_function_hooks("foo", 0, None)
    assert post_hook.call_count == 1


def test_remove_handle():
    hook_storage = HookStorage()

    handle1 = hook_storage.register_pre_function_hook("foo", 0, nn.Identity())
    handle2 = hook_storage.register_pre_function_hook("foo", 0, nn.Identity())
    assert len(hook_storage.pre_hooks["foo__0"]) == 2

    handle1.remove()
    assert len(hook_storage.pre_hooks["foo__0"]) == 1

    handle2.remove()
    assert "foo__0" not in hook_storage.pre_hooks


class CheckPriority(nn.Module):
    def __init__(self, storage: List, name: str):
        super().__init__()
        self.storage = storage
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.storage.append(self.name)
        return x


def test_excitation_priority():
    hook_storage = HookStorage()
    call_stack = []
    handles = []

    for group_name in ["1", "2", "3"]:
        hook = CheckPriority(call_stack, group_name)
        h = hook_storage.register_pre_function_hook("foo", 0, hook)
        handles.append(h)

    hook_storage.execute_pre_function_hooks("foo", 0, None)
    assert call_stack == ["1", "2", "3"]
    call_stack.clear()

    handles[1].remove()
    hook_storage.execute_pre_function_hooks("foo", 0, None)
    assert call_stack == ["1", "3"]
    call_stack.clear()

    hook_storage.register_pre_function_hook("foo", 0, CheckPriority(call_stack, "4"))
    hook_storage.execute_pre_function_hooks("foo", 0, None)
    assert call_stack == ["1", "3", "4"]


def test_named_hooks():
    hook_storage = HookStorage()
    hook1 = nn.Identity()
    hook2 = nn.Sequential(nn.Identity(), nn.Identity())

    hook_storage.register_pre_function_hook("foo", 0, hook1)
    hook_storage.register_pre_function_hook("foo", 0, hook1)

    hook_storage.register_post_function_hook("foo", 0, hook2)

    ret = list(hook_storage.named_hooks())
    ref = [("pre_hooks.foo__0.0", hook1), ("post_hooks.foo__0.0", hook2)]
    assert ret == ref

    ret = list(hook_storage.named_hooks(remove_duplicate=False))
    ref = [("pre_hooks.foo__0.0", hook1), ("pre_hooks.foo__0.1", hook1), ("post_hooks.foo__0.0", hook2)]
    assert ret == ref

    ret = list(hook_storage.named_hooks("pr", remove_duplicate=False))
    ref = [("pr.pre_hooks.foo__0.0", hook1), ("pr.pre_hooks.foo__0.1", hook1), ("pr.post_hooks.foo__0.0", hook2)]
    assert ret == ref


@pytest.mark.parametrize(
    "hook_name, ref",
    (
        ("pre_hooks.foo__0.0", ("pre_hooks", "foo", 0)),
        ("post_hooks.foo__1.0", ("post_hooks", "foo", 1)),
        ("__nncf_hooks.pre_hooks.foo__0.0", ("pre_hooks", "foo", 0)),
        ("__nncf_hooks.post_hooks.foo__1.0", ("post_hooks", "foo", 1)),
        ("post_hooks.conv:weight__0.0", ("post_hooks", "conv.weight", 0)),
        ("__nncf_hooks.post_hooks.conv:weight__0.0", ("post_hooks", "conv.weight", 0)),
    ),
)
def test_unwrap_hook_name(hook_name: str, ref: tuple[str, str, int]):
    assert unwrap_hook_name(hook_name) == ref


@pytest.mark.parametrize(
    "hook_name",
    (
        "foo__0.0",
        "pre_hooks.foo__0",
        "pre_hooks.foo.0",
        "pre.foo__0.0",
    ),
)
def test_unwrap_hook_name_raise_error(hook_name: str):
    with pytest.raises(ValueError, match="Invalid hook name"):
        unwrap_hook_name(hook_name)
