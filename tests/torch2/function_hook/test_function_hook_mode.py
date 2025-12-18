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


from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional, Union

import pytest
import torch
from pytest import FixtureRequest
from torch import nn

from nncf.torch.function_hook.hook_executor_mode import FunctionHookMode
from nncf.torch.function_hook.hook_executor_mode import OpMeta
from nncf.torch.function_hook.hook_executor_mode import generate_normalized_op_name
from nncf.torch.function_hook.hook_storage import HookStorage
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import register_pre_function_hook
from nncf.torch.function_hook.wrapper import wrap_model
from tests.torch2.function_hook import helpers
from tests.torch2.function_hook.helpers import CallCount
from tests.torch2.function_hook.helpers import CounterHook
from tests.torch2.function_hook.helpers import SharedParamModel


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

    hook_executor_mode.push_module_call_stack(hook_storage.post_hooks["conv/conv2d/0__0"]["0"])
    assert hook_executor_mode.get_current_relative_name() == "conv/post_hook__conv-conv2d-0__0[0]"


def test_get_current_executed_op_name():
    model = helpers.get_wrapped_simple_model_with_hook()
    hook_storage = get_hook_storage(model)
    hook_executor_mode = FunctionHookMode(model, hook_storage)

    hook_executor_mode.push_module_call_stack(model)
    assert hook_executor_mode.get_current_executed_op_name("foo") == "/foo/0"
    assert hook_executor_mode.get_current_executed_op_name("foo") == "/foo/1"

    hook_executor_mode.push_module_call_stack(model.conv)
    assert hook_executor_mode.get_current_executed_op_name("foo") == "conv/foo/0"
    assert hook_executor_mode.get_current_executed_op_name("foo") == "conv/foo/1"

    hook_executor_mode.push_module_call_stack(hook_storage.post_hooks["conv/conv2d/0__0"]["0"])
    assert hook_executor_mode.get_current_executed_op_name("foo") == "conv/post_hook__conv-conv2d-0__0[0]/foo/0"


NamedTuple = namedtuple("NamedTuple", ["output1", "output2"])


@pytest.fixture(params=["tensor", "list", "torch_return_type", "named_tuple"])
def example_outputs(
    request: FixtureRequest,
) -> Union[torch.Tensor, list[torch.Tensor], torch.return_types.max, NamedTuple]:
    return {
        "tensor": torch.tensor(1),
        "list": [torch.tensor(1), torch.tensor([2])],
        "torch_return_type": torch.return_types.max((torch.tensor(1), torch.tensor([2]))),
        "named_tuple": NamedTuple(torch.tensor(1), torch.tensor([2])),
    }.get(request.param)


def test_execute_post_hooks(
    example_outputs: Union[torch.Tensor, list[torch.Tensor], torch.return_types.max, NamedTuple],
):
    op_name = "/relu/0"
    hook_storage = HookStorage()
    hook_port_0 = CallCount()
    hook_port_1 = CallCount()
    hook_storage.register_post_function_hook(op_name, 0, hook_port_0)
    hook_storage.register_post_function_hook(op_name, 1, hook_port_1)
    ctx = FunctionHookMode(nn.Identity(), hook_storage)
    op_meta = OpMeta("/relu/0", torch.relu)
    ret_val = ctx.execute_post_hooks(example_outputs, op_meta)
    assert type(example_outputs) is type(ret_val)
    assert example_outputs == ret_val

    assert hook_port_0.call_count == 1
    if isinstance(example_outputs, torch.Tensor):
        assert hook_port_1.call_count == 0
    else:
        assert hook_port_1.call_count == 1


def test_process_model_output(
    example_outputs: Union[torch.Tensor, list[torch.Tensor], torch.return_types.max, NamedTuple],
):
    hook_storage = HookStorage()
    hook_output = CallCount()
    hook_output_0 = CallCount()
    hook_output_1 = CallCount()
    hook_storage.register_pre_function_hook("output", 0, hook_output)
    hook_storage.register_pre_function_hook("output_0", 0, hook_output_0)
    hook_storage.register_pre_function_hook("output_1", 0, hook_output_1)

    ctx = FunctionHookMode(nn.Identity(), hook_storage)
    ret_val = ctx.process_model_outputs(example_outputs)
    assert type(example_outputs) is type(ret_val)
    assert example_outputs == ret_val
    if isinstance(example_outputs, torch.Tensor):
        assert hook_output.call_count == 1
    else:
        assert hook_output_0.call_count == 1
        assert hook_output_1.call_count == 1


class ConcatModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x], dim=0)


class AddModule(nn.Module):
    def __init__(self, val: int) -> None:
        super().__init__()
        self.arg = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.arg


def test_execute_pre_hooks_for_concat():
    model = wrap_model(ConcatModel())
    op_name = "/cat/0"
    register_pre_function_hook(model, op_name, 0, AddModule(1))
    register_pre_function_hook(model, op_name, 1, AddModule(2))
    ret_val = model(torch.zeros(2))
    assert torch.allclose(ret_val, torch.tensor([1.0, 1.0, 2.0, 2.0])), ret_val


def test_shared_parameters():
    model = SharedParamModel()
    hook_storage = HookStorage()
    hook = CounterHook()
    hook_storage.register_post_function_hook("module1:0:weight", 0, hook)

    args = (model.get_example_inputs(),)
    kwargs = {}
    with FunctionHookMode(model, hook_storage) as ctx:
        assert hook.counter == 0
        assert ctx.cache_parameters == {}
        assert ctx.counter_reusing_shared_weights == {id(model.module1[0].weight): 1}

        args, kwargs = ctx.process_model_inputs(args, kwargs)
        outputs = model.forward(*args, **kwargs)
        outputs = ctx.process_model_outputs(outputs)

    assert hook.counter == 1
    # Check that the cache cleared in the end
    assert ctx.cache_parameters == {}
    assert ctx.counter_reusing_shared_weights == {}
