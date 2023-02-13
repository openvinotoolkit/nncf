import torch

from nncf.torch.dynamic_graph.patch_pytorch import MagicFunctionsToPatch
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta

from tests.shared.isolation_runner import run_pytest_case_function_in_separate_process
from tests.torch.pytorch_patch_isolated import test_jit_if_tracing_script_source_equals


def test_get_all_aliases_is_valid():
    operator_names_to_function_name = {}
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        operator_names_to_function_name[operator] = PT_OPERATOR_METATYPES.get(operator).get_all_aliases()

    invalid_metatypes = []
    for operator_metatypes, function_names in operator_names_to_function_name.items():
        if not function_names:
            invalid_metatypes.append(operator_metatypes)
    assert not invalid_metatypes, \
        f'There are metatypes with invalid `get_all_aliaces` method: {invalid_metatypes}'


def test_are_all_magic_functions_patched():
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        for function_name in PT_OPERATOR_METATYPES.get(operator).get_all_aliases():
            if function_name.startswith('__') and function_name.endswith('__'):
                is_contained = False
                for _, functions in MagicFunctionsToPatch.MAGIC_FUNCTIONS_TO_PATCH.items():
                    if function_name in functions:
                        is_contained = True
                        break
                assert is_contained


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
        return torch.empty(x.shape)

    class TestModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return test_fn(x)

    # ONNX export should work correctly because torch.jit.script_if_tracing is patched
    torch.onnx.export(TestModel(), (torch.zeros((1,)),), str(tmp_path / "jit_if_tracing_test_model.onnx"))


def test_jit_if_tracing_script_source():
    # Run test case in a separate process to track patching of torch by NNCF
    run_pytest_case_function_in_separate_process(test_jit_if_tracing_script_source_equals)
