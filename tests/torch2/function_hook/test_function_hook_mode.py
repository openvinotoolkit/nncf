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

from nncf.experimental.torch2.function_hook.hook_executor_mode import FunctionHookMode
from nncf.experimental.torch2.function_hook.hook_executor_mode import generate_normalized_op_name
from nncf.experimental.torch2.function_hook.hook_storage import HookType
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from tests.torch2.function_hook import helpers


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
    ),
    ids=idfn,
)
def test_generate_normalized_op_name(param: NormalizedOpNameTestCase):
    op_name = generate_normalized_op_name(module_name=param.model_name, fn_name=param.fn_name, call_id=param.call_id)
    assert op_name == param.ref


def test_current_relative_name():
    model = helpers.get_wrapped_simple_model_with_hook()
    hook_storage = get_hook_storage(model)
    hook_executor_mode = FunctionHookMode(model, hook_storage)
    hook_executor_mode.push_module_call_stack(model)
    # assert hook_executor_mode.get_current_relative_name() == ""

    hook_executor_mode.push_module_call_stack(model.conv)
    # assert hook_executor_mode.get_current_relative_name() == "conv"

    hook_executor_mode.push_module_call_stack(hook_storage.storage[f"{HookType.POST_HOOK}__conv/conv2d/0__0"]["0"])
    assert hook_executor_mode.get_current_relative_name() == "conv/post_hook__conv-conv2d-0__0[0]"


def test_get_current_executed_op_name():
    model = helpers.get_wrapped_simple_model_with_hook()
    hook_storage = get_hook_storage(model)
    hook_executor_mode = FunctionHookMode(model, hook_storage)

    hook_executor_mode.push_module_call_stack(model)
    assert hook_executor_mode.get_current_executed_op_name("foo") == "/foo/0"
    hook_executor_mode.register_op("foo")
    assert hook_executor_mode.get_current_executed_op_name("foo") == "/foo/1"

    hook_executor_mode.push_module_call_stack(model.conv)
    assert hook_executor_mode.get_current_executed_op_name("foo") == "conv/foo/0"
    hook_executor_mode.register_op("foo")
    assert hook_executor_mode.get_current_executed_op_name("foo") == "conv/foo/1"

    hook_executor_mode.push_module_call_stack(hook_storage.storage[f"{HookType.POST_HOOK}__conv/conv2d/0__0"]["0"])
    assert hook_executor_mode.get_current_executed_op_name("foo") == "conv/post_hook__conv-conv2d-0__0[0]/foo/0"
