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


import pytest
import torch
from torch import nn

from nncf.torch.function_hook.hook_storage import HookStorage
from nncf.torch.function_hook.hook_storage import decode_hook_name
from tests.torch.function_hook.helpers import CallCount


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
    def __init__(self, storage: list, name: str):
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
def test_decode_hook_name(hook_name: str, ref: tuple[str, str, int]):
    assert decode_hook_name(hook_name) == ref


INVALID_HOOK_NAMES = [
    "foo__0.0",
    "pre_hooks.foo__0",
    "pre_hooks.foo.0",
    "pre.foo__0.0",
]


@pytest.mark.parametrize("hook_name", INVALID_HOOK_NAMES)
def test_decode_hook_name_raise_error(hook_name: str):
    with pytest.raises(ValueError, match="Invalid hook name"):
        decode_hook_name(hook_name)


def test_can_delete_hook():
    hook_storage = HookStorage()
    hook1 = nn.Identity()
    hook2 = nn.Sequential(nn.Identity(), nn.Identity())

    hook_storage.register_pre_function_hook("foo", 0, hook1)
    hook_storage.register_pre_function_hook("foo", 0, hook1)
    hook_storage.register_post_function_hook("foo", 0, hook2)
    hook_storage.register_post_function_hook("foo", 0, hook1)
    assert list(hook_storage.named_hooks(remove_duplicate=False)) == [
        ("pre_hooks.foo__0.0", hook1),
        ("pre_hooks.foo__0.1", hook1),
        ("post_hooks.foo__0.0", hook2),
        ("post_hooks.foo__0.1", hook1),
    ]

    hook_storage.delete_hook("pre_hooks.foo__0.0")
    ret = list(hook_storage.named_hooks(remove_duplicate=False))
    ref = [("pre_hooks.foo__0.1", hook1), ("post_hooks.foo__0.0", hook2), ("post_hooks.foo__0.1", hook1)]
    assert ret == ref

    hook_storage.delete_hook("post_hooks.foo__0.0")
    ret = list(hook_storage.named_hooks(remove_duplicate=False))
    ref = [("pre_hooks.foo__0.1", hook1), ("post_hooks.foo__0.1", hook1)]
    assert ret == ref


not_existing_port = 1
not_existing_hook_id = 2


@pytest.mark.parametrize(
    ("hook_name", "match_msg"),
    [
        *((hook_name, "Invalid hook name") for hook_name in INVALID_HOOK_NAMES),
        (f"pre_hooks.foo__{not_existing_port}.0", "No hook was found"),
        (f"pre_hooks.foo__0.{not_existing_hook_id}", "No hook was found"),
    ],
)
def test_can_not_delete_hook_with_invalid_args(hook_name, match_msg):
    hook_storage = HookStorage()
    hook1 = nn.Identity()
    hook2 = nn.Sequential(nn.Identity(), nn.Identity())

    hook_storage.register_pre_function_hook("foo", 0, hook1)
    hook_storage.register_pre_function_hook("foo", 0, hook1)
    hook_storage.register_post_function_hook("foo", 0, hook2)
    hook_storage.register_post_function_hook("foo", 0, hook1)

    with pytest.raises(ValueError, match=match_msg):
        hook_storage.delete_hook(hook_name)


def test_can_not_delete_hook_twice():
    hook_storage = HookStorage()
    hook1 = nn.Identity()

    hook_storage.register_pre_function_hook("foo", 0, hook1)
    hook_storage.delete_hook("pre_hooks.foo__0.0")
    ret = list(hook_storage.named_hooks(remove_duplicate=False))
    assert ret == []
    with pytest.raises(ValueError, match="No hook was found"):
        hook_storage.delete_hook("pre_hooks.foo__0.0")


def test_is_empty():
    hook_storage = HookStorage()
    hook1 = nn.Identity()
    assert hook_storage.is_empty()

    handle = hook_storage.register_pre_function_hook("foo", 0, hook1)
    assert not hook_storage.is_empty()
    handle.remove()
    assert hook_storage.is_empty()

    handle = hook_storage.register_post_function_hook("foo", 0, hook1)
    assert not hook_storage.is_empty()
    handle.remove()
    assert hook_storage.is_empty()
