# Copyright (c) 2024 Intel Corporation
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

from nncf.torch2.hook_executor.hook_storage import HookStorage
from nncf.torch2.hook_executor.hook_storage import HookType
from tests.torch2.hook_executor.helpers import CallCount


class CheckPriority(nn.Module):
    def __init__(self, storage: List, name: str):
        super().__init__()
        self.storage = storage
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.storage.append(self.name)
        return x


@pytest.fixture(params=HookType)
def hook_type(request: pytest.FixtureRequest):
    return request.param


def test_insert(hook_type: HookType):
    hooks = HookStorage()
    module = nn.Identity()
    hooks.insert_hook("group", hook_type, "foo", 0, module)
    assert hooks.storage["group"][f"{hook_type.value}__foo__0"] is module


def test_execute(hook_type: HookType):
    hooks = HookStorage()
    module = CallCount()
    hooks.insert_hook("group", hook_type, "foo", 0, module)
    assert hooks.storage["group"][f"{hook_type.value}__foo__0"] is module
    hooks.execute_hook(hook_type, "foo", 0, torch.tensor(1))
    assert module.call_count == 1


@pytest.mark.parametrize(
    "group_names",
    (
        ("5", "1", "01"),
        ("1", "2", "3"),
        ("2", "1", "9"),
    ),
)
def test_execute_priority(hook_type: HookType, group_names: List[str]):
    hooks = HookStorage()
    call_stack = []
    for group_name in group_names:
        module = CheckPriority(call_stack, group_name)
        hooks.insert_hook(group_name, hook_type, "foo", 0, module)
    hooks.execute_hook(hook_type, "foo", 0, torch.tensor(1))
    assert call_stack == sorted(group_names)


def test_remove_group(hook_type: HookType):
    hooks = HookStorage()
    module = CallCount()
    hooks.insert_hook("group", hook_type, "foo", 0, module)
    assert hooks.storage["group"][f"{hook_type.value}__foo__0"] is module
    hooks.execute_hook(hook_type, "foo", 0, torch.tensor(1))
    assert module.call_count == 1
    hooks.remove_group("group")
    with pytest.raises(KeyError):
        hooks.storage["group"][f"{hook_type.value}__foo__0"]
    hooks.execute_hook(hook_type, "foo", 0, torch.tensor(1))
    assert module.call_count == 1
