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

import inspect
import os
from typing import List

import pytest
import torch

import nncf
from nncf.common.utils.os import is_windows
from nncf.config import NNCFConfig
from nncf.torch import wrap_model
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.patch_pytorch import _ORIG_JIT_SCRIPT
from nncf.torch.dynamic_graph.patch_pytorch import MagicFunctionsToPatch
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from nncf.torch.dynamic_graph.patch_pytorch_state import PATCHING_STATE
from nncf.torch.dynamic_graph.structs import NamespaceTarget
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES
from tests.cross_fw.shared.isolation_runner import run_pytest_case_function_in_separate_process
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.pytorch_patch_isolated import test_compile
from tests.torch.pytorch_patch_isolated import test_jit_if_tracing_script_source_equals
from tests.torch.pytorch_patch_isolated import test_jit_script_exception_preserves_patching_isolated


def test_get_all_aliases_is_valid():
    operator_names_to_function_name = {}
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        operator_names_to_function_name[operator] = PT_OPERATOR_METATYPES.get(operator).get_all_aliases()

    invalid_metatypes = []
    for operator_metatypes, function_names in operator_names_to_function_name.items():
        if not function_names:
            invalid_metatypes.append(operator_metatypes)
    assert not invalid_metatypes, f"There are metatypes with invalid `get_all_aliases` method: {invalid_metatypes}"


@pytest.mark.parametrize("name_space", [NamespaceTarget.TORCH_TENSOR])
def test_patch_magic_functions(name_space):
    patched_magic_fns = MagicFunctionsToPatch.MAGIC_FUNCTIONS_TO_PATCH.get(name_space, [])
    for op_name, operator in PT_OPERATOR_METATYPES.registry_dict.items():
        op_fns: List[str] = operator.module_to_function_names.get(name_space, [])
        op_magic_fns = [x for x in op_fns if x.startswith("__") and x.endswith("__")]
        for fn_name in op_magic_fns:
            assert fn_name in patched_magic_fns, f"{op_name} contains not patched magic function {fn_name}"


@pytest.mark.parametrize("name_space", [NamespaceTarget.TORCH_TENSOR])
def test_op_for_patch_magic_functions(name_space):
    patched_magic_fns = MagicFunctionsToPatch.MAGIC_FUNCTIONS_TO_PATCH.get(name_space, [])

    all_magic_fns_in_op: List[str] = []
    for operator in PT_OPERATOR_METATYPES.registry_dict.values():
        op_fns: List[str] = operator.module_to_function_names.get(name_space, [])
        all_magic_fns_in_op += [x for x in op_fns if x.startswith("__") and x.endswith("__")]

    for patched_fn in patched_magic_fns:
        assert patched_fn in dir(torch.Tensor), f"Magic function {patched_fn} does not exist in Tensor"
        assert patched_fn in all_magic_fns_in_op, f"No metatype for patched magic function {patched_fn}"


def test_tensor_printing_does_not_inflate_graph():
    context_to_use = TracingContext()
    context_to_use.enable_trace_dynamic_graph()
    with context_to_use as _ctx:
        with torch.no_grad():
            tensor = torch.ones([1, 2])
            print(tensor)
            str(tensor)
            tensor.__repr__()
            tensor = TracedTensor.from_torch_tensor(tensor, TensorMeta(0, 0, tensor.shape))
            print(tensor)
            str(tensor)
            tensor.__repr__()
    assert _ctx.graph.get_nodes_count() == 0


def test_jit_if_tracing_script_patching(tmp_path):
    @torch.jit.script_if_tracing
    def test_fn(x: torch.Tensor):
        return torch.ones(x.shape)

    class TestModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return test_fn(x)

    # ONNX export should work correctly because torch.jit.script_if_tracing is patched
    torch.onnx.export(TestModel(), (torch.zeros((1,)),), str(tmp_path / "jit_if_tracing_test_model.onnx"))


def test_jit_if_tracing_script_source():
    # Run test case in a separate process to track patching of torch by NNCF
    run_pytest_case_function_in_separate_process(test_jit_if_tracing_script_source_equals)


def test_jit_script_exception_preserves_patching():
    # Run test case in a separate process to track patching of torch by NNCF
    run_pytest_case_function_in_separate_process(test_jit_script_exception_preserves_patching_isolated)


@pytest.mark.xfail(is_windows(), reason="https://github.com/pytorch/pytorch/issues/122094")
@pytest.mark.parametrize("compile_forward", [False, True])
def test_torch_compile(compile_forward):
    # Run test case in a separate process to track patching of torch by NNCF
    os.environ["COMPILE_FORWARD"] = f"{int(compile_forward)}"
    run_pytest_case_function_in_separate_process(test_compile)


def test_torch_compile_on_nncf_model():
    # Calling torch.compile on a regular torch model should work fine
    model = BasicConvTestModel()
    compiled_model = torch.compile(model)
    compiled_model(torch.ones(model.INPUT_SIZE))

    model = BasicConvTestModel()
    quantized_model = nncf.quantize(model, nncf.Dataset([torch.rand(model.INPUT_SIZE)]))
    with pytest.raises(
        TypeError, match="At the moment torch\\.compile\\(\\) is not supported for models optimized by NNCF\\."
    ):
        torch.compile(quantized_model)

    model = BasicConvTestModel()
    config = get_test_quantization_config(model)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    with pytest.raises(
        TypeError, match="At the moment torch\\.compile\\(\\) is not supported for models optimized by NNCF\\."
    ):
        torch.compile(compressed_model)

    stripped_model = compression_ctrl.strip()
    with pytest.raises(
        TypeError, match="At the moment torch\\.compile\\(\\) is not supported for models optimized by NNCF\\."
    ):
        torch.compile(stripped_model)

    with pytest.raises(
        TypeError, match="At the moment torch\\.compile\\(\\) is not supported for models optimized by NNCF\\."
    ):
        # Compiling this model would actually work, but inference of the compiled model will fail
        torch.compile(model)


def test_jit_script_signature():
    # Check that torch.jit.script has the same signature as the wrapper was designed for
    signature = inspect.signature(_ORIG_JIT_SCRIPT)
    assert "obj" in signature.parameters and "_rcb" in signature.parameters and "_frames_up" in signature.parameters


def test_jit_script_class():
    # Define an outside function to test custom resolution callback inside torch_jit_script_wrapper
    def outside_function(x):
        return x + torch.tensor(1.0)

    class TestClass:
        def class_method(self, x):
            return outside_function(x)

    # Scripting a class instead of a method to trigger custom resolution callback usage
    torch.jit.script(TestClass)


def test_jit_trace_model():
    model = BasicConvTestModel()
    config = get_test_quantization_config(model)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    torch.jit.trace(compressed_model, example_inputs=torch.rand(model.INPUT_SIZE))

    model = compression_ctrl.strip()
    torch.jit.trace(model, example_inputs=torch.rand(model.INPUT_SIZE))


def get_test_quantization_config(model):
    config = NNCFConfig()
    config.update(
        {
            "model": "model",
            "input_info": {"sample_size": model.INPUT_SIZE},
            "compression": {"algorithm": "quantization"},
        }
    )
    register_bn_adaptation_init_args(config)
    return config


def test_operator_unpatching():
    unwrapped_operator_expected = False

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.op = torch.nn.functional.gelu

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            ctx = get_current_context()
            original_in_operator_value = ctx.in_operator
            ctx.in_operator = "unexpected_value"
            result = self.op(x)
            # If ctx.in_operator value was not changed, then operator wrapper exited early
            assert not unwrapped_operator_expected or ctx.in_operator == "unexpected_value"
            ctx.in_operator = original_in_operator_value
            return result

    test_model = TestModel()
    example_input = torch.ones((1,))
    wrapped_operator = wrap_model(test_model, example_input)
    with disable_patching():
        assert not PATCHING_STATE.operators_are_wrapped
        # Despite patching is disabled, test_model holds a reference to a patched operator
        assert "_original_op" in test_model.op.__dict__
        unwrapped_operator_expected = True
        wrapped_operator(example_input)
