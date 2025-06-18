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

import types
from copy import deepcopy
from functools import partial
from pathlib import Path

import onnxruntime as ort
import pytest
import torch

from nncf.torch.function_hook.wrapper import is_wrapped
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.function_hook.wrapper import register_pre_function_hook
from nncf.torch.function_hook.wrapper import wrap_model
from tests.torch2.function_hook import helpers

ADD_VALUE = 2.0


@pytest.mark.parametrize("forward_type", ["origin", "partial", "bound", "fn"])
def test_wrapper(forward_type: str):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    model.eval()

    model._old_forward = model.forward

    if forward_type == "partial":
        # Like in accelerate module
        def new_forward(self, x):
            return self._old_forward(x)

        model.forward = partial(new_forward, model)

    elif forward_type == "methodtype":

        def new_forward(self, x):
            return model._old_forward(x)

        model.forward = types.MethodType(new_forward, model)
    elif forward_type == "fn":
        old_forward = model.forward

        def new_forward(x):
            return old_forward(x)

        model.forward = new_forward

    with torch.no_grad():
        ret = model(example_input)
        wrapped = wrap_model(model)
        wrapped_ret = wrapped(example_input)

    torch.testing.assert_close(ret, wrapped_ret)


def test_export_strict_false():
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)

    wrapped = wrap_model(model)
    register_post_function_hook(wrapped, "/relu/0", 0, helpers.AddModule(ADD_VALUE))
    reference = wrapped(example_input)

    m_traced = torch.export.export(wrapped, args=(example_input,), strict=False)
    actual = m_traced.module()(example_input)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual, return_origin + ADD_VALUE)


def test_jit_trace():
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)

    wrapped = wrap_model(model)
    register_post_function_hook(wrapped, "/relu/0", 0, helpers.AddModule(ADD_VALUE))
    reference = wrapped(example_input)

    m_traced = torch.jit.trace(wrapped, example_inputs=(example_input,), strict=False)
    actual = m_traced(example_input)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual, return_origin + ADD_VALUE)


def test_compile_via_trace():
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)
    wrapped = wrap_model(model)
    register_post_function_hook(wrapped, "/relu/0", 0, helpers.AddModule(ADD_VALUE))
    reference = wrapped(example_input)
    m_traced = torch.jit.trace(wrapped, example_inputs=(example_input,), strict=False)
    m_compiled = torch.compile(m_traced)
    actual = m_compiled(example_input)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual, return_origin + ADD_VALUE)


def test_export_onnx(tmp_path: Path):
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)

    wrapped = wrap_model(model)
    register_post_function_hook(wrapped, "/relu/0", 0, helpers.AddModule(ADD_VALUE))
    reference = wrapped(example_input)

    onnx_file = tmp_path / "model.onnx"
    torch.onnx.export(wrapped, (example_input,), onnx_file.as_posix())
    session = ort.InferenceSession(onnx_file)

    actual = session.run(None, {"input": example_input.numpy()})[0]
    torch.testing.assert_close(torch.tensor(actual), reference)
    torch.testing.assert_close(torch.tensor(actual), return_origin + ADD_VALUE)


def test_deepcopy():
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.get_wrapped_simple_model_with_hook()
    ref = model(example_input)
    copy = deepcopy(model)
    del model
    import gc

    gc.collect()
    act = copy(example_input)
    torch.testing.assert_close(act, ref)


def test_torch_save_load(tmp_path: Path):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.get_wrapped_simple_model_with_hook()
    ref = model(example_input)
    path = tmp_path / "model.pt"
    torch.save(model, path)
    del model
    import gc

    gc.collect()
    loaded = torch.load(path, weights_only=False)
    act = loaded(example_input)
    torch.testing.assert_close(act, ref)


@pytest.mark.parametrize(
    "hook_type, target_name",
    (
        ("post_hook", "x"),
        ("pre_hook", "/relu/0"),
        ("post_hook", "/relu/0"),
        ("pre_hook", "output"),
        ("post_hook", "conv:weight"),
    ),
)
def test_insert_hook(hook_type, target_name):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    wrapped = wrap_model(model)
    assert is_wrapped(wrapped)

    hook = helpers.CallCount()
    if hook_type == "pre_hook":
        register_pre_function_hook(wrapped, target_name, 0, hook)
    else:
        register_post_function_hook(wrapped, target_name, 0, hook)

    wrapped(example_input)
    assert hook.call_count == 1


@pytest.mark.parametrize("hook_type", ["pre_hook", "post_hook"])
def test_insert_nested_hook(hook_type: str):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    wrapped = wrap_model(model)

    hook = helpers.CallCount()
    if hook_type == "pre_hook":
        register_pre_function_hook(wrapped, "/relu/0", 0, helpers.AddModule(2.0))
        register_pre_function_hook(wrapped, "pre_hook__-relu-0__0[0]/add/0", 0, hook)
    else:
        register_post_function_hook(wrapped, "/relu/0", 0, helpers.AddModule(2.0))
        register_post_function_hook(wrapped, "post_hook__-relu-0__0[0]/add/0", 0, hook)
    wrapped(example_input)

    assert hook.call_count == 1
