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


from dataclasses import dataclass
from typing import Any, Optional

import pytest

from nncf.torch2.hook_executor.hook_executor_mode import generate_normalized_op_name
from nncf.torch2.hook_executor.hook_storage import HookType
from nncf.torch2.hook_executor.wrapper import insert_hook
from nncf.torch2.hook_executor.wrapper import is_wrapped
from nncf.torch2.hook_executor.wrapper import wrap_model
from tests.torch2.hook_executor import helpers


@dataclass
class NormalizedOpNameTestCase:
    model_name: str
    fn_name: str
    call_id: Optional[str]
    ref: str


def idfn(value: Any):
    if isinstance(value, NormalizedOpNameTestCase):
        return value.ref
    return None


@pytest.mark.parametrize(
    "param",
    (
        NormalizedOpNameTestCase("module", "foo", None, "module/foo"),
        NormalizedOpNameTestCase("module", "foo", 0, "module/foo/0"),
        NormalizedOpNameTestCase("module", "foo", 1, "module/foo/1"),
        NormalizedOpNameTestCase("module.module", "foo", None, "module:module/foo"),
        NormalizedOpNameTestCase("module.module", "foo", 0, "module:module/foo/0"),
        NormalizedOpNameTestCase("hook.module:module/mod", "foo", None, "hook:module:module-mod/foo"),
    ),
    ids=idfn,
)
def test_generate_normalized_op_name(param: NormalizedOpNameTestCase):
    op_name = generate_normalized_op_name(module_name=param.model_name, fn_name=param.fn_name, call_id=param.call_id)
    assert op_name == param.ref


@pytest.mark.parametrize(
    "hook_type, target_name",
    (
        (HookType.POST_HOOK, "x"),
        (HookType.PRE_HOOK, "/relu/0"),
        (HookType.POST_HOOK, "/relu/0"),
        (HookType.PRE_HOOK, "output"),
        (HookType.POST_HOOK, "conv:weight"),
    ),
)
def test_insert_hook(hook_type, target_name):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    wrapped = wrap_model(model)
    assert is_wrapped(wrapped)

    hook = helpers.CallCount()
    insert_hook(wrapped, "hook_group", hook_type, target_name, 0, hook)
    wrapped(example_input)
    assert hook.call_count == 1


@pytest.mark.parametrize("hook_type", HookType)
def test_insert_hook_twice_raise(hook_type):
    model = helpers.ConvModel()
    wrapped = wrap_model(model)

    hook = helpers.CallCount()
    insert_hook(wrapped, "hook_group", hook_type, "/relu/0", 0, hook)
    with pytest.raises(RuntimeError, match="Hook already set for.*"):
        insert_hook(wrapped, "hook_group", hook_type, "/relu/0", 0, hook)


@pytest.mark.parametrize("hook_type", HookType)
def test_insert_nested_hook(hook_type: HookType):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    wrapped = wrap_model(model)

    hook = helpers.CallCount()
    insert_hook(wrapped, "hook_group", hook_type, "/relu/0", 0, helpers.AddModule(2.0))
    insert_hook(wrapped, "hook_group", hook_type, f"[hook_group:{hook_type.value}__-relu-0__0]/add/0", 0, hook)
    wrapped(example_input)

    assert hook.call_count == 1
